"""Sync benchmark results from logs into experiments.md.

Rules:
- Pick the highest-step benchmark found for each EXP section.
- Match against a canonical baseline with the exact same step horizon only.
- Insert or replace an auto-generated block inside the experiment section.
- Never replace an existing auto block with one that has fewer or equal steps.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from statistics import fmean


ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT, "benchmark_logs")
EXPERIMENTS_MD = os.path.join(ROOT, "experiments.md")

SECTION_HEADER_RE = re.compile(
    r"^\s*\[(\d+)/(\d+)\]\s+(.+?)(?:\s+\((\d+)\s+steps\))?(?:\s+~.*)?\s*$"
)
STEP_RE = re.compile(
    r"step\s+(\d+).*?loss:\s+([\d.]+).*?tok/sec:\s+([\d,]+)",
    re.IGNORECASE,
)
FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
EXP_ID_RE = re.compile(r"\bEXP-(\d{3})(?:\.(\d+))?\b", re.IGNORECASE)
DOC_SECTION_RE = re.compile(r"^(##|###) EXP-(\d{3})(?:\.(\d+))?: .*$", re.MULTILINE)


@dataclass
class LogSection:
    log_path: str
    log_name: str
    section: str
    steps: int | None
    exp_id: str | None
    val_bpb: float | None
    training_seconds: float | None
    avg_toksec: float | None
    step_losses: dict[int, float]
    metric_rows: list[dict[str, float]]
    mtime: float


def is_canonical_baseline(name: str) -> bool:
    lowered = name.strip().lower()
    return lowered == "baseline" or lowered.startswith("baseline ")


def baseline_priority(name: str) -> int:
    lowered = name.strip().lower()
    return 0 if lowered == "baseline" else 1


def parse_section_header(line: str):
    match = SECTION_HEADER_RE.search(line)
    if not match:
        return None
    return {
        "name": match.group(3).strip(),
        "steps": int(match.group(4)) if match.group(4) else None,
    }


def parse_metric_line(line: str):
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
        if match:
            metrics[key.strip()] = float(match.group(0))
    return metrics or None


def parse_log(filepath: str) -> list[LogSection]:
    sections = []
    current = None
    toksec_values = []
    step_losses = {}
    metric_rows = []

    def finalize_current():
        nonlocal current, toksec_values, step_losses, metric_rows
        if current is None:
            return
        exp_match = EXP_ID_RE.search(current["section"])
        sections.append(
            LogSection(
                log_path=filepath,
                log_name=os.path.basename(filepath),
                section=current["section"],
                steps=current["steps"],
                exp_id=exp_match.group(1) if exp_match else None,
                val_bpb=current["val_bpb"],
                training_seconds=current["training_seconds"],
                avg_toksec=fmean(toksec_values) if toksec_values else None,
                step_losses=dict(step_losses),
                metric_rows=list(metric_rows),
                mtime=os.path.getmtime(filepath),
            )
        )
        current = None
        toksec_values = []
        step_losses = {}
        metric_rows = []

    with open(filepath, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            header = parse_section_header(line)
            if header:
                finalize_current()
                current = {
                    "section": header["name"],
                    "steps": header["steps"],
                    "val_bpb": None,
                    "training_seconds": None,
                }
                continue

            if current is None:
                continue

            step_match = STEP_RE.search(line)
            if step_match:
                step = int(step_match.group(1))
                loss = float(step_match.group(2))
                toksec = float(step_match.group(3).replace(",", ""))
                step_losses[step] = loss
                if toksec > 100:
                    toksec_values.append(toksec)
                continue

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
                metric_rows.append(metrics)

    finalize_current()
    return sections


def gather_sections() -> list[LogSection]:
    rows = []
    for filepath in sorted(glob.glob(os.path.join(LOG_DIR, "bench_*.log"))):
        rows.extend(parse_log(filepath))
    return rows


def choose_best_experiment_sections(sections: list[LogSection]) -> dict[str, LogSection]:
    best = {}
    for section in sections:
        if not section.exp_id:
            continue
        current = best.get(section.exp_id)
        if current is None:
            best[section.exp_id] = section
            continue
        current_steps = current.steps if current.steps is not None else -1
        section_steps = section.steps if section.steps is not None else -1
        if section_steps > current_steps:
            best[section.exp_id] = section
        elif section_steps == current_steps and section.mtime > current.mtime:
            best[section.exp_id] = section
    return best


def choose_baseline(section: LogSection, sections: list[LogSection]) -> LogSection | None:
    candidates = [row for row in sections if is_canonical_baseline(row.section)]
    if not candidates:
        return None

    same_log_exact = [
        row for row in candidates
        if row.log_path == section.log_path and row.steps == section.steps
    ]
    if same_log_exact:
        return max(same_log_exact, key=lambda row: row.mtime)

    exact = [row for row in candidates if row.steps == section.steps]
    if exact:
        return min(exact, key=lambda row: (baseline_priority(row.section), -row.mtime))
    return None


def find_baseline_step(target_loss: float, bl_steps: list[int], bl_losses: list[float]) -> float:
    if target_loss >= bl_losses[0]:
        return 0.0
    if target_loss <= bl_losses[-1]:
        if len(bl_steps) >= 2 and bl_losses[-2] > bl_losses[-1]:
            slope = (bl_losses[-1] - bl_losses[-2]) / (bl_steps[-1] - bl_steps[-2])
            return bl_steps[-1] + (target_loss - bl_losses[-1]) / slope
        return float(bl_steps[-1])
    for index in range(len(bl_losses) - 1):
        if bl_losses[index] >= target_loss >= bl_losses[index + 1]:
            frac = (bl_losses[index] - target_loss) / (bl_losses[index] - bl_losses[index + 1])
            return bl_steps[index] + frac * (bl_steps[index + 1] - bl_steps[index])
    return float(bl_steps[-1])


def compute_lag_summary(exp_section: LogSection, baseline_section: LogSection):
    if not exp_section.step_losses or not baseline_section.step_losses:
        return None
    exp_steps = sorted(exp_section.step_losses)
    bl_steps = sorted(baseline_section.step_losses)
    exp_final_step = exp_steps[-1]
    exp_final_loss = exp_section.step_losses[exp_final_step]
    bl_final_loss = baseline_section.step_losses.get(exp_final_step)
    bl_losses = [baseline_section.step_losses[step] for step in bl_steps]
    lag_final = exp_final_step - find_baseline_step(exp_final_loss, bl_steps, bl_losses)
    diff = None if bl_final_loss is None else exp_final_loss - bl_final_loss
    diff_pct = None if bl_final_loss in (None, 0) else (diff / bl_final_loss) * 100
    return {
        "final_step": exp_final_step,
        "exp_final_loss": exp_final_loss,
        "bl_final_loss": bl_final_loss,
        "diff": diff,
        "diff_pct": diff_pct,
        "lag_final": lag_final,
    }


def summarize_metrics(section: LogSection) -> dict[str, float | None]:
    keys = ["survival", "head_effrank", "rank_ratio", "top10e", "cos", "union_rank"]
    summary = {}
    for key in keys:
        values = [row[key] for row in section.metric_rows if key in row]
        summary[key] = fmean(values) if values else None
    return summary


def format_metric_pair(label: str, exp_value, bl_value, digits: int = 3) -> str | None:
    if exp_value is None:
        return None
    exp_str = f"{exp_value:.{digits}f}"
    if bl_value is None:
        return f"`{label}` {exp_str}"
    delta = exp_value - bl_value
    delta_str = f"{delta:+.{digits}f}"
    return f"`{label}` {exp_str} vs {bl_value:.{digits}f} ({delta_str})"


def classify_outcome(exp_section: LogSection, baseline_section: LogSection | None, lag_summary) -> str:
    if baseline_section is None:
        return "N/A"
    if exp_section.val_bpb is None or baseline_section.val_bpb is None:
        return "PENDING"
    delta = exp_section.val_bpb - baseline_section.val_bpb
    if delta <= -0.005:
        return "BETTER"
    if delta >= 0.005:
        return "WORSE"
    if lag_summary and lag_summary["diff"] is not None:
        if lag_summary["diff"] < 0:
            return "SLIGHTLY BETTER"
        if lag_summary["diff"] > 0:
            return "SLIGHTLY WORSE"
    return "MIXED"


def build_auto_block(exp_section: LogSection, baseline_section: LogSection | None) -> str:
    lag = compute_lag_summary(exp_section, baseline_section) if baseline_section is not None else None
    exp_metrics = summarize_metrics(exp_section)
    bl_metrics = summarize_metrics(baseline_section) if baseline_section is not None else {}
    outcome = classify_outcome(exp_section, baseline_section, lag)
    synced_on = "2026-03-17"

    lines = [
        f"<!-- AUTO-RESULTS EXP-{exp_section.exp_id} START steps={exp_section.steps or 0} -->",
        f"**Auto-benchmark sync ({exp_section.steps or '?'} steps, synced {synced_on})**",
        "",
        f"- Source log: `benchmark_logs/{exp_section.log_name}` -> `{exp_section.section}`",
    ]

    if baseline_section is None:
        lines.append("- Matched baseline: `N/A` (no exact same-step canonical baseline found)")
    else:
        lines.append(f"- Matched baseline: `benchmark_logs/{baseline_section.log_name}` -> `{baseline_section.section}`")

    if baseline_section is not None and exp_section.val_bpb is not None and baseline_section.val_bpb is not None:
        delta = exp_section.val_bpb - baseline_section.val_bpb
        pct = (delta / baseline_section.val_bpb) * 100 if baseline_section.val_bpb else 0.0
        lines.append(
            f"- val_bpb: `{exp_section.val_bpb:.6f}` vs `{baseline_section.val_bpb:.6f}` (`{delta:+.6f}`, `{pct:+.2f}%`)"
        )
    elif exp_section.val_bpb is not None:
        lines.append(f"- val_bpb: `{exp_section.val_bpb:.6f}` vs `N/A`")

    if lag and lag["bl_final_loss"] is not None and lag["diff"] is not None and lag["diff_pct"] is not None:
        lines.append(
            f"- Final loss (step {lag['final_step']}): `{lag['exp_final_loss']:.4f}` vs `{lag['bl_final_loss']:.4f}` (`{lag['diff']:+.4f}`, `{lag['diff_pct']:+.2f}%`)"
        )
        lines.append(f"- Final lag: `{lag['lag_final']:+.1f}` baseline-equivalent steps")
    elif exp_section.step_losses:
        final_step = max(exp_section.step_losses)
        lines.append(f"- Final loss (step {final_step}): `{exp_section.step_losses[final_step]:.4f}` vs `N/A`")

    if baseline_section is not None and exp_section.avg_toksec is not None and baseline_section.avg_toksec is not None:
        tps_delta = exp_section.avg_toksec - baseline_section.avg_toksec
        tps_pct = (tps_delta / baseline_section.avg_toksec) * 100 if baseline_section.avg_toksec else 0.0
        lines.append(
            f"- Throughput: `{exp_section.avg_toksec:,.0f}` tok/s vs `{baseline_section.avg_toksec:,.0f}` (`{tps_pct:+.1f}%`)"
        )
    elif exp_section.avg_toksec is not None:
        lines.append(f"- Throughput: `{exp_section.avg_toksec:,.0f}` tok/s vs `N/A`")

    metric_pairs = [
        format_metric_pair("surv", exp_metrics["survival"], bl_metrics.get("survival"), 3),
        format_metric_pair("eff", exp_metrics["head_effrank"], bl_metrics.get("head_effrank"), 1),
        format_metric_pair("rr", exp_metrics["rank_ratio"], bl_metrics.get("rank_ratio"), 3),
        format_metric_pair("top10e", exp_metrics["top10e"], bl_metrics.get("top10e"), 1),
        format_metric_pair("cos", exp_metrics["cos"], bl_metrics.get("cos"), 4),
        format_metric_pair("uni", exp_metrics["union_rank"], bl_metrics.get("union_rank"), 1),
    ]
    metric_pairs = [item for item in metric_pairs if item]
    if metric_pairs:
        lines.append(f"- Diagnostics: {'; '.join(metric_pairs)}")

    lines.append(f"- Auto outcome: `{outcome}` against matched baseline")
    lines.append(f"<!-- AUTO-RESULTS EXP-{exp_section.exp_id} END -->")
    return "\n".join(lines)


def get_existing_auto_steps(section_body: str, exp_id: str) -> int | None:
    match = re.search(
        rf"<!-- AUTO-RESULTS EXP-{exp_id} START steps=(\d+) -->",
        section_body,
    )
    if not match:
        return None
    return int(match.group(1))


def replace_section_auto_block(section_body: str, exp_id: str, new_block: str) -> str:
    pattern = re.compile(
        rf"\n*<!-- AUTO-RESULTS EXP-{exp_id} START steps=\d+ -->.*?<!-- AUTO-RESULTS EXP-{exp_id} END -->\n*",
        re.S,
    )
    body_without_auto = pattern.sub("\n\n", section_body).lstrip("\n")
    return f"\n\n{new_block}\n\n{body_without_auto}"


def sync_document(text: str, best_sections: dict[str, LogSection], all_sections: list[LogSection], selected_ids: set[str] | None):
    matches = list(DOC_SECTION_RE.finditer(text))
    updated = text
    replacements = []

    for index, match in enumerate(matches):
        exp_id = match.group(2)
        if selected_ids and exp_id not in selected_ids:
            continue
        exp_section = best_sections.get(exp_id)
        if exp_section is None:
            continue

        baseline_section = choose_baseline(exp_section, all_sections)

        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_body = text[start:end]
        existing_steps = get_existing_auto_steps(section_body, exp_id)
        new_steps = exp_section.steps or 0
        if existing_steps is not None and existing_steps >= new_steps:
            continue

        new_body = replace_section_auto_block(section_body, exp_id, build_auto_block(exp_section, baseline_section))
        replacements.append((start, end, new_body, exp_section, baseline_section))

    for start, end, new_body, _, _ in reversed(replacements):
        updated = updated[:start] + new_body + updated[end:]
    return updated, replacements


def build_parser():
    parser = argparse.ArgumentParser(description="Sync benchmark results into experiments.md.")
    parser.add_argument("--doc", default=EXPERIMENTS_MD, help="Path to experiments.md")
    parser.add_argument("--apply", action="store_true", help="Write changes to the document")
    parser.add_argument("--exp", action="append", help="Limit sync to a specific EXP id, e.g. 012 or EXP-012")
    return parser


def normalize_selected_ids(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    selected = set()
    for value in values:
        match = EXP_ID_RE.search(value)
        if match:
            selected.add(match.group(1))
        else:
            selected.add(value.zfill(3))
    return selected


def main():
    args = build_parser().parse_args()
    with open(args.doc, encoding="utf-8") as handle:
        text = handle.read()

    all_sections = gather_sections()
    best_sections = choose_best_experiment_sections(all_sections)
    selected_ids = normalize_selected_ids(args.exp)
    updated_text, replacements = sync_document(text, best_sections, all_sections, selected_ids)

    if not replacements:
        print("Nenhuma secao precisou ser atualizada.")
        return

    for _, _, _, exp_section, baseline_section in replacements:
        if baseline_section is None:
            print(
                f"EXP-{exp_section.exp_id}: {exp_section.section} ({exp_section.steps or '?'} steps) "
                f"<= baseline N/A"
            )
        else:
            print(
                f"EXP-{exp_section.exp_id}: {exp_section.section} ({exp_section.steps or '?'} steps) "
                f"<= baseline {baseline_section.section} @ {baseline_section.log_name}"
            )

    if args.apply:
        with open(args.doc, "w", encoding="utf-8") as handle:
            handle.write(updated_text)
        print(f"Atualizado: {args.doc}")
    else:
        print("Dry run concluido. Use --apply para gravar.")


if __name__ == "__main__":
    main()