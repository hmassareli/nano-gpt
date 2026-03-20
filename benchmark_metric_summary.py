"""Summarize gradient-metric sections across benchmark logs.

Scans benchmark logs, extracts sections that contain gradient metrics,
prints an aggregate sortable table, and can also print sampled metric rows
every N steps.
"""

import argparse
import glob
import os
import re
from statistics import fmean


LOG_DIR = "benchmark_logs"
SECTION_HEADER_RE = re.compile(
    r"^\s*\[(\d+)/(\d+)\]\s+(.+?)(?:\s+\((\d+)\s+steps\))?(?:\s+~.*)?\s*$"
)
TOKSEC_RE = re.compile(r"tok/sec:\s*([\d,]+)")
FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
STEP_RE = re.compile(
    r"step\s+(\d+).*?loss:\s+([\d.]+).*?lrm:\s+([\d.]+).*?tok/sec:\s+([\d,]+)",
    re.IGNORECASE,
)


def parse_section_header(line):
    match = SECTION_HEADER_RE.search(line)
    if not match:
        return None
    return {
        "index": int(match.group(1)),
        "total": int(match.group(2)),
        "name": match.group(3).strip(),
        "steps": int(match.group(4)) if match.group(4) else None,
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


def parse_step_line(line):
    match = STEP_RE.search(line)
    if not match:
        return None
    return {
        "step": int(match.group(1)),
        "loss": float(match.group(2)),
        "lrm": float(match.group(3)),
        "tok_sec": float(match.group(4).replace(",", "")),
    }


def safe_mean(values):
    return fmean(values) if values else None


def safe_max(values):
    return max(values) if values else None


def parse_log(filepath):
    sections = []
    current = None
    last_step = None

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
                    "metric_samples": [],
                    "toksec": [],
                    "val_bpb": None,
                    "training_seconds": None,
                }
                last_step = None
                continue

            if current is None:
                continue

            step_info = parse_step_line(line)
            if step_info:
                last_step = step_info
                current["toksec"].append(step_info["tok_sec"])
                continue

            toksec_match = TOKSEC_RE.search(line)
            if toksec_match and last_step is None:
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
                sample = dict(metrics)
                if last_step is not None:
                    sample.update(last_step)
                current["metric_samples"].append(sample)

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
        "metric_samples": section["metric_samples"],
    }

    canonical_names = [
        "backbone",
        "head",
        "survival",
        "head_effrank",
        "rank_ratio",
        "top10e",
        "head_drift0",
        "head_delta",
        "conf",
        "margin",
        "ent",
        "ent_ratio",
        "post_perturb",
        "perturb_strength",
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
        ("file", 14),
        ("section", 20),
        ("n", 3),
        ("bpb", 8),
        ("eff", 6),
        ("eff*", 6),
        ("rr", 5),
        ("rr*", 5),
        ("surv", 6),
        ("top10e", 7),
    ]

    if any(row["avg_union_rank"] is not None for row in display_rows):
        headers.append(("uni", 6))
    if any(row["avg_cos"] is not None for row in display_rows):
        headers.append(("cos", 7))
    if any(row["avg_head_drift0"] is not None for row in display_rows):
        headers.append(("drift0", 7))
    if any(row["avg_head_delta"] is not None for row in display_rows):
        headers.append(("delta", 7))
    if any(row["avg_conf"] is not None for row in display_rows):
        headers.append(("conf", 6))
    if any(row["avg_margin"] is not None for row in display_rows):
        headers.append(("margin", 7))
    if any(row["avg_ent_ratio"] is not None for row in display_rows):
        headers.append(("ent%", 6))

    headers.append(("tok/s", 7))

    separator = " | "
    print(separator.join(label.ljust(width) for label, width in headers))
    print(separator.join("-" * width for _, width in headers))
    for row in display_rows:
        values = [
            shorten(row["file"], 14),
            shorten(row["section"], 20),
            str(row["samples"]),
            format_value(row["val_bpb"], 6),
            format_value(row["avg_head_effrank"], 1),
            format_value(row["max_head_effrank"], 1),
            format_value(row["avg_rank_ratio"], 3),
            format_value(row["max_rank_ratio"], 3),
            format_value(row["avg_survival"], 3),
            format_value(row["avg_top10e"], 1),
        ]
        if any(label == "uni" for label, _ in headers):
            values.append(format_value(row["avg_union_rank"], 1))
        if any(label == "cos" for label, _ in headers):
            values.append(format_value(row["avg_cos"], 4))
        if any(label == "drift0" for label, _ in headers):
            values.append(format_value(row["avg_head_drift0"], 4))
        if any(label == "delta" for label, _ in headers):
            values.append(format_value(row["avg_head_delta"], 4))
        if any(label == "conf" for label, _ in headers):
            values.append(format_value(row["avg_conf"], 4))
        if any(label == "margin" for label, _ in headers):
            values.append(format_value(row["avg_margin"], 4))
        if any(label == "ent%" for label, _ in headers):
            values.append(format_value(row["avg_ent_ratio"], 4))
        values.append(format_value(row["avg_toksec"], 0))
        print(separator.join(value.ljust(width) for value, (_, width) in zip(values, headers)))


def print_details(rows):
    print("\nDetalhes das metricas presentes por secao:")
    for row in rows:
        print(f"- {row['file']} | {row['section']} | {row['metrics_present']}")


def sample_metric_rows(samples, every):
    if not samples:
        return []

    sampled = []
    for sample in samples:
        step = sample.get("step")
        if step is None:
            continue
        if step == 0 or step % every == 0:
            sampled.append(sample)

    last = samples[-1]
    if last.get("step") is not None and (not sampled or sampled[-1].get("step") != last.get("step")):
        sampled.append(last)
    return sampled


def print_samples(rows, every, limit):
    detail_rows = rows[:limit] if limit else rows
    for row in detail_rows:
        print(f"\n[{row['file']}] {row['section']}")
        print(
            f"steps={row['steps'] or '-'} | val_bpb={format_value(row['val_bpb'], 6)} | "
            f"training_s={format_value(row['training_seconds'], 1)}"
        )
        print(f"Sampling every {every} steps:")
        headers = [
            ("step", 6),
            ("loss", 7),
            ("lrm", 5),
            ("tok/s", 7),
            ("surv", 6),
            ("eff", 7),
            ("rr", 5),
            ("top10e", 7),
            ("uni", 7),
            ("cos", 7),
            ("drift0", 7),
            ("delta", 7),
            ("conf", 6),
            ("margin", 7),
            ("ent%", 6),
            ("pert", 5),
        ]
        separator = " | "
        print(separator.join(label.ljust(width) for label, width in headers))
        print(separator.join("-" * width for _, width in headers))
        for sample in sample_metric_rows(row["metric_samples"], every):
            values = [
                str(sample.get("step", "-")),
                format_value(sample.get("loss"), 4),
                format_value(sample.get("lrm"), 2),
                format_value(sample.get("tok_sec"), 0),
                format_value(sample.get("survival"), 3),
                format_value(sample.get("head_effrank"), 1),
                format_value(sample.get("rank_ratio"), 3),
                format_value(sample.get("top10e"), 1),
                format_value(sample.get("union_rank"), 1),
                format_value(sample.get("cos"), 4),
                format_value(sample.get("head_drift0"), 4),
                format_value(sample.get("head_delta"), 4),
                format_value(sample.get("conf"), 4),
                format_value(sample.get("margin"), 4),
                format_value(sample.get("ent_ratio"), 4),
                format_value(sample.get("post_perturb"), 0),
            ]
            print(separator.join(value.ljust(width) for value, (_, width) in zip(values, headers)))


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
            "avg_head_drift0",
            "avg_head_delta",
            "avg_conf",
            "avg_margin",
            "avg_ent_ratio",
            "avg_union_rank",
            "avg_toksec",
            "val_bpb",
        ],
        help="Column used to sort descending, except val_bpb which sorts ascending.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit displayed rows.")
    parser.add_argument(
        "--every",
        type=int,
        default=20,
        help="Sample detailed metric rows every N steps.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print metrics-present details for every included section.",
    )
    parser.add_argument(
        "--no-samples",
        action="store_true",
        help="Do not print sampled metric rows; only print the aggregate table.",
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

    if not args.no_samples:
        print_samples(rows, args.every, args.limit)

    if args.details:
        print_details(rows)


if __name__ == "__main__":
    main()