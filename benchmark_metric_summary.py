"""Summarize gradient-metric sections across benchmark logs.

Scans benchmark logs, extracts only sections that contain at least one
"grads |" line, and prints a sortable table with aggregate metrics.
"""

import argparse
import glob
import os
import re
from statistics import fmean


LOG_DIR = "benchmark_logs"
SECTION_HEADER_RE = re.compile(r"\[(\d+)/(\d+)\]\s+(.+?)\s+\((\d+)\s+steps\)")
TOKSEC_RE = re.compile(r"tok/sec:\s*([\d,]+)")
FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def parse_section_header(line):
    match = SECTION_HEADER_RE.search(line)
    if not match:
        return None
    return {
        "index": int(match.group(1)),
        "total": int(match.group(2)),
        "name": match.group(3).strip(),
        "steps": int(match.group(4)),
    }


def parse_metric_line(line):
    if "grads |" not in line:
        return None

    payload = line.split("grads |", 1)[1]
    metrics = {}
    for chunk in payload.split("|"):
        chunk = chunk.strip()
        if ":" not in chunk:
            continue
        key, raw_value = chunk.split(":", 1)
        match = FLOAT_RE.search(raw_value.replace(",", ""))
        if not match:
            continue
        metrics[key.strip()] = float(match.group(0))
    return metrics or None


def safe_mean(values):
    return fmean(values) if values else None


def safe_max(values):
    return max(values) if values else None


def parse_log(filepath):
    sections = []
    current = None

    with open(filepath, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")

            header = parse_section_header(line)
            if header:
                if current is not None:
                    sections.append(finalize_section(current))
                current = {
                    "file": os.path.basename(filepath),
                    "file_path": filepath,
                    "section": header["name"],
                    "steps": header["steps"],
                    "metric_rows": [],
                    "toksec": [],
                    "val_bpb": None,
                    "training_seconds": None,
                }
                continue

            if current is None:
                continue

            toksec_match = TOKSEC_RE.search(line)
            if toksec_match:
                current["toksec"].append(float(toksec_match.group(1).replace(",", "")))

            if line.startswith("val_bpb:"):
                match = FLOAT_RE.search(line)
                if match:
                    current["val_bpb"] = float(match.group(0))
                continue

            if line.startswith("training_seconds:"):
                match = FLOAT_RE.search(line)
                if match:
                    current["training_seconds"] = float(match.group(0))
                continue

            metrics = parse_metric_line(line)
            if metrics:
                current["metric_rows"].append(metrics)

    if current is not None:
        sections.append(finalize_section(current))

    return [section for section in sections if section["samples"] > 0]


def finalize_section(section):
    metric_names = sorted({key for row in section["metric_rows"] for key in row})
    values = {name: [row[name] for row in section["metric_rows"] if name in row] for name in metric_names}

    summary = {
        "file": section["file"],
        "file_path": section["file_path"],
        "section": section["section"],
        "steps": section["steps"],
        "samples": len(section["metric_rows"]),
        "metrics_present": ",".join(metric_names),
        "val_bpb": section["val_bpb"],
        "training_seconds": section["training_seconds"],
        "avg_toksec": safe_mean(section["toksec"]),
    }

    canonical_names = [
        "backbone",
        "head",
        "survival",
        "head_effrank",
        "rank_ratio",
        "top10e",
        "cos",
        "union_rank",
    ]
    for name in canonical_names:
        summary[f"avg_{name}"] = safe_mean(values.get(name, []))
        summary[f"max_{name}"] = safe_max(values.get(name, []))

    return summary


def gather_sections(pattern):
    rows = []
    for filepath in sorted(glob.glob(pattern)):
        rows.extend(parse_log(filepath))
    return rows


def infer_sort_value(row, sort_key):
    if sort_key == "best":
        for key in ("avg_rank_ratio", "max_rank_ratio", "avg_head_effrank", "max_head_effrank"):
            value = row.get(key)
            if value is not None:
                return value
        return float("-inf")
    value = row.get(sort_key)
    return float("-inf") if value is None else value


def format_value(value, digits=3):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def shorten(text, width):
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def print_table(rows, limit):
    display_rows = rows[:limit] if limit else rows
    headers = [
        ("file", 16),
        ("section", 24),
        ("samples", 7),
        ("val_bpb", 8),
        ("avg_eff", 7),
        ("max_eff", 7),
        ("avg_rr", 6),
        ("max_rr", 6),
        ("avg_surv", 8),
        ("avg_top10e", 10),
        ("avg_uni", 7),
        ("avg_cos", 8),
        ("avg_tok", 7),
    ]

    separator = " | "
    print(separator.join(label.ljust(width) for label, width in headers))
    print(separator.join("-" * width for _, width in headers))
    for row in display_rows:
        values = [
            shorten(row["file"], 16),
            shorten(row["section"], 24),
            str(row["samples"]),
            format_value(row["val_bpb"], 6),
            format_value(row["avg_head_effrank"], 1),
            format_value(row["max_head_effrank"], 1),
            format_value(row["avg_rank_ratio"], 3),
            format_value(row["max_rank_ratio"], 3),
            format_value(row["avg_survival"], 3),
            format_value(row["avg_top10e"], 1),
            format_value(row["avg_union_rank"], 1),
            format_value(row["avg_cos"], 4),
            format_value(row["avg_toksec"], 0),
        ]
        print(separator.join(value.ljust(width) for value, (_, width) in zip(values, headers)))


def print_details(rows):
    print("\nDetalhes das metricas presentes por secao:")
    for row in rows:
        print(f"- {row['file']} | {row['section']} | {row['metrics_present']}")


def build_parser():
    parser = argparse.ArgumentParser(description="Summarize gradient metrics from benchmark logs.")
    parser.add_argument(
        "--pattern",
        default=os.path.join(LOG_DIR, "bench_*.log"),
        help="Glob pattern for benchmark logs.",
    )
    parser.add_argument(
        "--sort",
        default="best",
        choices=[
            "best",
            "avg_head_effrank",
            "max_head_effrank",
            "avg_rank_ratio",
            "max_rank_ratio",
            "avg_survival",
            "avg_top10e",
            "avg_union_rank",
            "avg_toksec",
            "val_bpb",
        ],
        help="Column used to sort descending, except val_bpb which sorts ascending.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit displayed rows.")
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print metrics-present details for every included section.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    rows = gather_sections(args.pattern)

    if not rows:
        print("Nenhuma secao com metricas de gradiente encontrada.")
        return

    reverse = args.sort != "val_bpb"
    rows.sort(key=lambda row: infer_sort_value(row, args.sort), reverse=reverse)

    print(f"Secoes com metricas encontradas: {len(rows)}")
    print(f"Padrao analisado: {args.pattern}")
    print(f"Ordenacao: {args.sort} ({'desc' if reverse else 'asc'})\n")
    print_table(rows, args.limit)

    if args.details:
        print_details(rows)


if __name__ == "__main__":
    main()