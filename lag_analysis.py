"""EXP vs Baseline lag analysis — interactive CLI or batch CLI."""
import argparse
import glob
import os
import re
import sys

LOG_DIR = "benchmark_logs"
SECTION_HEADER_RE = re.compile(
    r"^\s*\[(\d+)/(\d+)\]\s+(.+?)(?:\s+\((\d+)\s+steps\))?(?:\s+~.*)?\s*$"
)


def parse_section_header(line):
    """Return the exact section name when the line is a benchmark section header."""
    info = parse_section_header_info(line)
    return info["name"] if info else None


def parse_section_header_info(line):
    """Return parsed section header info when the line is a benchmark section header."""
    match = SECTION_HEADER_RE.search(line)
    if not match:
        return None
    return {
        "index": int(match.group(1)),
        "count": int(match.group(2)),
        "name": match.group(3).strip(),
        "steps": int(match.group(4)) if match.group(4) else None,
    }


def is_canonical_baseline_section(section_name):
    """Return True only for plain baseline variants, not baseline-derived experiments."""
    lowered = section_name.strip().lower()
    return lowered == "baseline" or lowered.startswith("baseline ")


def list_logs():
    """Return sorted list of benchmark log files."""
    pattern = os.path.join(LOG_DIR, "bench_*.log")
    return sorted(glob.glob(pattern))


def list_sections(filepath):
    """Extract section names from a benchmark log."""
    return [info["name"] for info in list_sections_info(filepath)]


def list_sections_info(filepath):
    """Extract parsed section headers from a benchmark log."""
    sections = []
    with open(filepath, encoding="utf-8") as file_handle:
        for line in file_handle:
            info = parse_section_header_info(line)
            if info:
                sections.append(info)
    return sections


def get_section_steps(filepath, section_name):
    """Return the declared step horizon for a section when available."""
    for info in list_sections_info(filepath):
        if info["name"] == section_name:
            return info["steps"]
    return None


def resolve_log_path(log_value):
    """Resolve a log path from absolute path, relative path, or basename."""
    if not log_value:
        return None
    candidates = []
    if os.path.isabs(log_value):
        candidates.append(log_value)
    else:
        candidates.append(log_value)
        candidates.append(os.path.join(LOG_DIR, log_value))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Log nao encontrado: {log_value}")


def pick_latest_log():
    """Return the newest benchmark log path."""
    logs = list_logs()
    if not logs:
        raise FileNotFoundError(f"Nenhum log encontrado em {LOG_DIR}/")
    return max(logs, key=os.path.getmtime)


def resolve_section_name(sections, query, role):
    """Resolve a section by exact match or unique substring."""
    if not sections:
        raise ValueError(f"Nenhuma secao disponivel para {role}")
    if not query:
        raise ValueError(f"Secao de {role} nao informada")

    exact = [section for section in sections if section == query]
    if exact:
        return exact[0]

    lowered = query.lower()
    partial = [section for section in sections if lowered in section.lower()]
    if len(partial) == 1:
        return partial[0]
    if not partial:
        raise ValueError(f"Secao '{query}' nao encontrada para {role}")
    raise ValueError(f"Secao '{query}' ambigua para {role}: " + ", ".join(partial))


def find_default_baseline_section(sections):
    """Pick a baseline section automatically when possible."""
    baseline_like = [section for section in sections if is_canonical_baseline_section(section)]
    if len(baseline_like) == 1:
        return baseline_like[0]
    if len(baseline_like) > 1:
        exact = [section for section in baseline_like if section.strip().lower() == "baseline"]
        if exact:
            return exact[0]
    return None


def list_baseline_candidates():
    """Return canonical baseline sections across all logs."""
    candidates = []
    for log_path in list_logs():
        for info in list_sections_info(log_path):
            if is_canonical_baseline_section(info["name"]):
                candidates.append(
                    {
                        "log_path": log_path,
                        "log_name": os.path.basename(log_path),
                        "section": info["name"],
                        "steps": info["steps"],
                        "mtime": os.path.getmtime(log_path),
                    }
                )
    return candidates


def choose_best_baseline_for_steps(target_steps, preferred_log=None):
    """Pick the most appropriate baseline for an experiment step horizon."""
    candidates = list_baseline_candidates()
    if not candidates:
        raise ValueError("Nenhum baseline encontrado nos logs")

    if preferred_log:
        same_log_exact = [
            candidate
            for candidate in candidates
            if os.path.normcase(candidate["log_path"]) == os.path.normcase(preferred_log)
            and candidate["steps"] == target_steps
        ]
        if same_log_exact:
            return max(same_log_exact, key=lambda item: item["mtime"])

    exact = [candidate for candidate in candidates if candidate["steps"] == target_steps]
    if exact:
        return max(exact, key=lambda item: item["mtime"])

    with_steps = [candidate for candidate in candidates if candidate["steps"] is not None]
    if with_steps:
        return min(
            with_steps,
            key=lambda item: (abs(item["steps"] - target_steps), -item["mtime"]),
        )

    return max(candidates, key=lambda item: item["mtime"])


def parse_losses(filepath, section_keyword):
    """Parse step losses for a given section keyword."""
    losses = {}
    in_section = False
    with open(filepath, encoding="utf-8") as file_handle:
        for line in file_handle:
            section_name = parse_section_header(line)
            if section_name == section_keyword:
                in_section = True
                continue
            if in_section and section_name:
                break
            if in_section:
                match = re.search(r"step (\d+).*?loss: ([\d.]+)", line)
                if match:
                    losses[int(match.group(1))] = float(match.group(2))
                if line.strip() == "---" and len(losses) > 5:
                    break
    return losses


def parse_avg_toksec(filepath, section_keyword):
    """Return average tok/sec for a section, ignoring obvious outliers."""
    values = []
    in_section = False
    with open(filepath, encoding="utf-8") as file_handle:
        for line in file_handle:
            section_name = parse_section_header(line)
            if section_name == section_keyword:
                in_section = True
                continue
            if in_section and section_name:
                break
            if in_section:
                match = re.search(r"tok/sec:\s*([\d,]+)", line)
                if match:
                    value = int(match.group(1).replace(",", ""))
                    if value > 100:
                        values.append(value)
                if line.strip() == "---" and len(values) > 5:
                    break
    return sum(values) / len(values) if values else None


def iter_non_baseline_sections(sections, baseline_section):
    """Yield sections excluding the chosen baseline section."""
    for section in sections:
        if section != baseline_section:
            yield section


def find_baseline_step(target_loss, bl_steps, bl_losses):
    """Estimate which baseline step reaches the experiment's target loss."""
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


def pick_option(prompt, options):
    """Show numbered list and return chosen item."""
    print(f"\n{prompt}")
    for index, option in enumerate(options, 1):
        print(f"  [{index}] {option}")
    while True:
        try:
            choice = int(input("\n> "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
        except (ValueError, EOFError):
            pass
        print(f"  Escolha 1-{len(options)}")


def build_parser():
    parser = argparse.ArgumentParser(description="Lag analysis for benchmark logs.")
    parser.add_argument("--every", type=int, default=10, help="Show sampled rows every N steps. Default: 10.")
    parser.add_argument("--summary", action="store_true", help="Print a shorter report with fewer columns.")
    parser.add_argument("--log", help="Single log file to use for both baseline and experiment(s). Accepts basename or path.")
    parser.add_argument("--baseline-log", help="Log file containing the baseline section. Accepts basename or path.")
    parser.add_argument("--exp-log", help="Log file containing the experiment section. Accepts basename or path.")
    parser.add_argument("--baseline-section", help="Baseline section name or unique substring.")
    parser.add_argument("--exp-section", action="append", help="Experiment section name or unique substring. Can be repeated.")
    parser.add_argument("--compare-all", action="store_true", help="Compare every non-baseline section in the selected log against the baseline.")
    parser.add_argument("--latest", action="store_true", help="Use the latest benchmark log automatically.")
    return parser


def print_report(bl_path, bl_section, exp_path, exp_section, args):
    """Print one lag analysis report and return a process-like status code."""
    baseline = parse_losses(bl_path, bl_section)
    experiment = parse_losses(exp_path, exp_section)

    if not baseline:
        print(f"Sem dados de loss para '{bl_section}'")
        return 1
    if not experiment:
        print(f"Sem dados de loss para '{exp_section}'")
        return 1

    bl_steps = sorted(baseline.keys())
    bl_losses = [baseline[step] for step in bl_steps]
    exp_steps = sorted(experiment.keys())

    short_exp = exp_section.split(":")[0].strip() if ":" in exp_section else exp_section
    print(f"\n{short_exp} vs {bl_section}")
    print("=" * 80)

    window = 20
    sample_every = max(1, args.every)
    if args.summary:
        print(f"Sampling every {sample_every} steps (summary mode)")
        print(f"{'Step':>5} | {'EXP Loss':>9} | {'BL Loss':>9} | {'Diff%':>6} | {'Lag':>5}")
        print("-" * 47)
    else:
        print(f"Sampling every {sample_every} steps")
        print(f"{'Step':>5} | {'EXP Loss':>9} | {'BL Loss':>9} | {'Diff':>8} | {'Diff%':>6} | {'DrpΔ%':>6} | {'WinΔ%':>6} | {'Lag':>5} | Visual")
        print("-" * 98)

    exp_loss_0 = experiment[exp_steps[0]]
    bl_loss_0 = baseline[bl_steps[0]] if bl_steps else None

    all_diffs = []
    all_lags = []
    all_drop_deltas = []
    for position, step in enumerate(exp_steps):
        exp_loss = experiment[step]
        baseline_loss = baseline.get(step)
        equivalent_baseline_step = find_baseline_step(exp_loss, bl_steps, bl_losses)
        lag = step - equivalent_baseline_step
        all_lags.append(lag)

        if baseline_loss is not None and baseline_loss != 0:
            diff = exp_loss - baseline_loss
            diff_pct = (diff / baseline_loss) * 100
        else:
            diff = None
            diff_pct = None
        all_diffs.append((step, diff, diff_pct))

        drop_delta_pct = None
        if baseline_loss is not None and bl_loss_0 is not None:
            baseline_drop = bl_loss_0 - baseline_loss
            exp_drop = exp_loss_0 - exp_loss
            if baseline_drop > 0:
                drop_delta_pct = (exp_drop - baseline_drop) / baseline_drop * 100
        all_drop_deltas.append((step, drop_delta_pct))

        win_delta_pct = None
        if position >= window:
            step_back = exp_steps[position - window]
            exp_back = experiment[step_back]
            baseline_back = baseline.get(step_back)
            if baseline_loss is not None and baseline_back is not None:
                baseline_window_drop = baseline_back - baseline_loss
                exp_window_drop = exp_back - exp_loss
                if baseline_window_drop > 0:
                    win_delta_pct = (exp_window_drop - baseline_window_drop) / baseline_window_drop * 100

        if step != exp_steps[0] and step != exp_steps[-1] and step % sample_every != 0:
            continue

        baseline_str = f"{baseline_loss:>9.4f}" if baseline_loss is not None else f"{'—':>9}"
        diff_str = f"{diff:>+8.4f}" if diff is not None else f"{'—':>8}"
        pct_str = f"{diff_pct:>+6.1f}" if diff_pct is not None else f"{'—':>6}"
        drop_str = f"{drop_delta_pct:>+6.1f}" if drop_delta_pct is not None else f"{'—':>6}"
        win_str = f"{win_delta_pct:>+6.1f}" if win_delta_pct is not None else f"{'—':>6}"

        if args.summary:
            print(f"{step:>5} | {exp_loss:>9.4f} | {baseline_str} | {pct_str} | {lag:>+5.1f}")
        else:
            if diff_pct is not None:
                bar_len = int(round(abs(diff_pct)))
                if bar_len == 0:
                    bar = "|"
                elif diff_pct > 0:
                    bar = "+" * min(bar_len, 30)
                else:
                    bar = "-" * min(bar_len, 30)
            else:
                bar = ""
            print(f"{step:>5} | {exp_loss:>9.4f} | {baseline_str} | {diff_str} | {pct_str} | {drop_str} | {win_str} | {lag:>+5.1f} | {bar}")

    print("-" * (47 if args.summary else 98))

    valid = [(step, diff, pct) for step, diff, pct in all_diffs if diff is not None and step >= 10]
    if valid:
        last_step, last_diff, last_pct = valid[-1]
        best_diff = min(valid, key=lambda item: item[1])
        worst_diff = max(valid, key=lambda item: item[1])
        avg_diff = sum(diff for _, diff, _ in valid) / len(valid)
        print(f"Diff final (step {last_step:>3}):  {last_diff:+.4f}  ({last_pct:+.1f}%)")
        print(f"Diff media (step 10+):  {avg_diff:+.4f}")
        print(f"Melhor diff:            {best_diff[1]:+.4f}  (step {best_diff[0]})")
        print(f"Pior diff:              {worst_diff[1]:+.4f}  (step {worst_diff[0]})")

    valid_drops = [(step, diff) for step, diff in all_drop_deltas if diff is not None and step >= 10]
    if valid_drops:
        last_drop_step, last_drop = valid_drops[-1]
        print(f"Drop acum. EXP vs BL:   {last_drop:+.1f}%  (step {last_drop_step}) — (+)=EXP derrubou mais")

    lags_10 = [lag for step, lag in zip(exp_steps, all_lags) if step >= 10]
    if lags_10:
        print(f"Lag final:              {all_lags[-1]:+.1f} steps")

    exp_tps = parse_avg_toksec(exp_path, exp_section)
    bl_tps = parse_avg_toksec(bl_path, bl_section)
    if exp_tps and bl_tps:
        pct = (exp_tps - bl_tps) / bl_tps * 100
        sign = "+" if pct >= 0 else ""
        print(f"\nThroughput: EXP {exp_tps:,.0f} tok/s vs BL {bl_tps:,.0f} tok/s ({sign}{pct:.1f}%)")

    print()
    print("(+) = EXP loss MAIOR que BL (pior)")
    print("(-) = EXP loss MENOR que BL (melhor)")
    return 0


def run_interactive(args):
    print("=" * 50)
    print("  LAG ANALYSIS — Experimento vs Baseline")
    print("=" * 50)

    logs = list_logs()
    if not logs:
        print(f"Nenhum log encontrado em {LOG_DIR}/")
        return 1

    log_names = [os.path.basename(path) for path in logs]
    bl_name = pick_option("Arquivo com o BASELINE:", log_names)
    bl_path = os.path.join(LOG_DIR, bl_name)

    bl_sections = list_sections(bl_path)
    if not bl_sections:
        print("Nenhuma secao encontrada nesse log!")
        return 1
    bl_section = pick_option("Secao do BASELINE:", bl_sections)

    exp_name = pick_option("Arquivo com o EXPERIMENTO:", log_names)
    exp_path = os.path.join(LOG_DIR, exp_name)

    exp_sections = list_sections(exp_path)
    if not exp_sections:
        print("Nenhuma secao encontrada nesse log!")
        return 1
    exp_section = pick_option("Secao do EXPERIMENTO:", exp_sections)

    return print_report(bl_path, bl_section, exp_path, exp_section, args)


def run_non_interactive(args):
    if args.latest:
        shared_log = pick_latest_log()
    else:
        shared_log = resolve_log_path(args.log) if args.log else None

    fixed_baseline_log = resolve_log_path(args.baseline_log) if args.baseline_log else None
    exp_path = resolve_log_path(args.exp_log) if args.exp_log else shared_log
    if not exp_path:
        raise ValueError("Informe --log, --exp-log ou --latest")

    exp_sections = list_sections(exp_path)
    if not exp_sections:
        raise ValueError(f"Nenhuma secao encontrada em {os.path.basename(exp_path)}")

    exp_queries = args.exp_section or []
    if args.compare_all:
        if shared_log and exp_path == shared_log:
            same_log_baseline = find_default_baseline_section(exp_sections)
            if same_log_baseline:
                exp_names = list(iter_non_baseline_sections(exp_sections, same_log_baseline))
            else:
                exp_names = [section for section in exp_sections if not is_canonical_baseline_section(section)]
        else:
            exp_names = [section for section in exp_sections if not is_canonical_baseline_section(section)]
    else:
        if not exp_queries:
            raise ValueError("Informe --exp-section ou use --compare-all")
        exp_names = [resolve_section_name(exp_sections, query, "experimento") for query in exp_queries]

    comparisons = []
    for exp_name in exp_names:
        exp_steps = get_section_steps(exp_path, exp_name)
        chosen_bl_path = fixed_baseline_log
        chosen_bl_section = None

        if chosen_bl_path:
            chosen_bl_sections = list_sections(chosen_bl_path)
            if not chosen_bl_sections:
                raise ValueError(f"Nenhuma secao encontrada em {os.path.basename(chosen_bl_path)}")
            baseline_query = args.baseline_section or find_default_baseline_section(chosen_bl_sections)
            if not baseline_query:
                raise ValueError("Nao consegui inferir o baseline fixo. Informe --baseline-section")
            chosen_bl_section = resolve_section_name(chosen_bl_sections, baseline_query, "baseline")
        else:
            same_log_baseline = find_default_baseline_section(exp_sections)
            if same_log_baseline:
                same_log_steps = get_section_steps(exp_path, same_log_baseline)
                if exp_steps is None or same_log_steps is None or exp_steps == same_log_steps:
                    chosen_bl_path = exp_path
                    chosen_bl_section = same_log_baseline

        if chosen_bl_section is None:
            if exp_steps is None:
                raise ValueError(
                    f"Nao consegui inferir baseline para '{exp_name}' sem step horizon; informe --baseline-log/--baseline-section"
                )
            match = choose_best_baseline_for_steps(exp_steps, preferred_log=exp_path)
            chosen_bl_path = match["log_path"]
            chosen_bl_section = match["section"]

        comparisons.append(
            {
                "exp_log_path": exp_path,
                "exp_section": exp_name,
                "exp_steps": exp_steps,
                "bl_log_path": chosen_bl_path,
                "bl_section": chosen_bl_section,
            }
        )

    print("=" * 50)
    print("  LAG ANALYSIS — Batch mode")
    print("=" * 50)
    print(f"Experiment log:   {os.path.basename(exp_path)}")
    if fixed_baseline_log:
        print(f"Baseline source:  {os.path.basename(fixed_baseline_log)} (fixo)")
    else:
        print("Baseline source:  auto-match por step horizon")
    print(f"Comparisons:      {len(comparisons)}")

    exit_code = 0
    for index, comparison in enumerate(comparisons, 1):
        exp_steps_str = comparison["exp_steps"] if comparison["exp_steps"] is not None else "?"
        print(f"\n[{index}/{len(comparisons)}] {comparison['exp_section']} ({exp_steps_str} steps)")
        print(f"Baseline escolhido: {comparison['bl_section']} @ {os.path.basename(comparison['bl_log_path'])}")
        rc = print_report(
            comparison["bl_log_path"],
            comparison["bl_section"],
            comparison["exp_log_path"],
            comparison["exp_section"],
            args,
        )
        if rc != 0:
            exit_code = rc
    return exit_code


def main():
    args = build_parser().parse_args()
    non_interactive = any(
        [
            args.log,
            args.baseline_log,
            args.exp_log,
            args.baseline_section,
            args.exp_section,
            args.compare_all,
            args.latest,
        ]
    )
    try:
        if non_interactive:
            return run_non_interactive(args)
        return run_interactive(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Erro: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
