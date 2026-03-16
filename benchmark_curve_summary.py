"""Summarize benchmark loss curves by section.

Reads benchmark logs, extracts sections and step lines, and prints a compact
table sampling the curve every N steps. Useful for quickly comparing runs
without opening the full log or using the interactive lag analysis.
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
STEP_RE = re.compile(r"step\s+(\d+).*?loss:\s+([\d.]+).*?tok/sec:\s+([\d,]+)")
FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def parse_section_header(line):
    match = SECTION_HEADER_RE.search(line)
    if not match:
        return None
    steps = int(match.group(4)) if match.group(4) else None
    return {
        "index": int(match.group(1)),
        "total": int(match.group(2)),
        "name": match.group(3).strip(),
        "steps": steps,
    }


def parse_log(filepath):
    sections = []
    current = None

    with open(filepath, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")

            header = parse_section_header(line)
            if header:
                if current is not None:
                    sections.append(current)
                current = {
                    "file": os.path.basename(filepath),
                    "section": header["name"],
                    "declared_steps": header["steps"],
                    "step_data": [],
                    "val_bpb": None,
                    "training_seconds": None,
                }
                continue

            if current is None:
                continue

            step_match = STEP_RE.search(line)
            if step_match:
                current["step_data"].append({
                    "step": int(step_match.group(1)),
                    "loss": float(step_match.group(2)),
                    "tok_sec": int(step_match.group(3).replace(",", "")),
                })
                continue

            if line.startswith("val_bpb:"):
                value = FLOAT_RE.search(line)
                current["val_bpb"] = float(value.group(0)) if value else None
                continue

            if line.startswith("training_seconds:"):
                value = FLOAT_RE.search(line)
                current["training_seconds"] = float(value.group(0)) if value else None

    if current is not None:
        sections.append(current)

    return [section for section in sections if section["step_data"]]


def gather_sections(pattern):
    rows = []
    for filepath in sorted(glob.glob(pattern)):
        rows.extend(parse_log(filepath))
    return rows


def sample_steps(step_data, every, include_last=True):
    sampled = []
    for row in step_data:
        step = row["step"]
        if step == 0 or step % every == 0:
            sampled.append(row)
    if include_last and sampled and sampled[-1]["step"] != step_data[-1]["step"]:
        sampled.append(step_data[-1])
    elif include_last and not sampled:
        sampled.append(step_data[-1])
    return sampled


def format_value(value, digits=4):
    return "-" if value is None else f"{value:.{digits}f}"


def shorten(text, width):
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def print_overview(rows):
    print("Resumo das secoes encontradas:")
    print("file             | section                  | steps | loss_i  | loss_f  | avg_tok/s | val_bpb")
    print("---------------- | ------------------------ | ----- | ------- | ------- | --------- | --------")
    for row in rows:
        step_data = row["step_data"]
        avg_tok = fmean(item["tok_sec"] for item in step_data)
        print(
            f"{shorten(row['file'], 16):16} | {shorten(row['section'], 24):24} | "
            f"{step_data[-1]['step']:>5} | {step_data[0]['loss']:>7.4f} | {step_data[-1]['loss']:>7.4f} | "
            f"{avg_tok:>9.0f} | {format_value(row['val_bpb'], 6):>8}"
        )


def print_section_samples(row, every):
    print(f"\n[{row['file']}] {row['section']}")
    print(f"steps={row['step_data'][-1]['step']} | val_bpb={format_value(row['val_bpb'], 6)} | training_s={format_value(row['training_seconds'], 1)}")
    print(f"Sampling every {every} steps:")
    print("step   | loss    | tok/s")
    print("------ | ------- | -------")
    for item in sample_steps(row["step_data"], every):
        print(f"{item['step']:>6} | {item['loss']:>7.4f} | {item['tok_sec']:>7}")


def build_parser():
    parser = argparse.ArgumentParser(description="Summarize benchmark curves by sampling every N steps.")
    parser.add_argument(
        "--pattern",
        default=os.path.join(LOG_DIR, "bench_*.log"),
        help="Glob pattern for benchmark logs.",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=10,
        help="Sample every N steps.",
    )
    parser.add_argument(
        "--file-contains",
        default="",
        help="Only include logs whose filename contains this substring.",
    )
    parser.add_argument(
        "--section-contains",
        default="",
        help="Only include sections whose name contains this substring.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of sections printed in detail.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    rows = gather_sections(args.pattern)

    if args.file_contains:
        needle = args.file_contains.lower()
        rows = [row for row in rows if needle in row["file"].lower()]
    if args.section_contains:
        needle = args.section_contains.lower()
        rows = [row for row in rows if needle in row["section"].lower()]

    if not rows:
        print("Nenhuma secao com curva de loss encontrada.")
        return

    print_overview(rows)

    detail_rows = rows[: args.limit] if args.limit else rows
    for row in detail_rows:
        print_section_samples(row, args.every)


if __name__ == "__main__":
    main()