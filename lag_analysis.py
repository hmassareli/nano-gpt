"""EXP vs Baseline lag analysis — interactive CLI."""
import argparse
import re
import glob
import os

LOG_DIR = "benchmark_logs"
SECTION_HEADER_RE = re.compile(
    r'^\s*\[(\d+)/(\d+)\]\s+(.+?)(?:\s+\((\d+)\s+steps\))?(?:\s+~.*)?\s*$'
)

# ── helpers ──────────────────────────────────────────────

def parse_section_header(line):
    """Return the exact section name when the line is a benchmark section header."""
    m = SECTION_HEADER_RE.search(line)
    return m.group(3).strip() if m else None

def list_logs():
    """Return sorted list of benchmark log files."""
    pattern = os.path.join(LOG_DIR, "bench_*.log")
    return sorted(glob.glob(pattern))

def list_sections(filepath):
    """Extract section names from a benchmark log (e.g. 'EXP-005: Contrastive...')."""
    sections = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            section_name = parse_section_header(line)
            if section_name:
                sections.append(section_name)
    return sections

def parse_losses(filepath, section_keyword):
    """Parse step losses for a given section keyword."""
    losses = {}
    in_section = False
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            section_name = parse_section_header(line)
            if section_name == section_keyword:
                in_section = True
                continue
            if in_section and section_name:
                break  # next section started
            if in_section:
                m = re.search(r'step (\d+).*?loss: ([\d.]+)', line)
                if m:
                    losses[int(m.group(1))] = float(m.group(2))
                if line.strip() == '---' and len(losses) > 5:
                    break
    return losses

def parse_avg_toksec(filepath, section_keyword):
    """Return average tok/sec for a section (ignoring outliers like dt>1min)."""
    vals = []
    in_section = False
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            section_name = parse_section_header(line)
            if section_name == section_keyword:
                in_section = True
                continue
            if in_section and section_name:
                break
            if in_section:
                m = re.search(r'tok/sec:\s*([\d,]+)', line)
                if m:
                    v = int(m.group(1).replace(',', ''))
                    if v > 100:  # filter out outliers (e.g. sleep/pause)
                        vals.append(v)
                if line.strip() == '---' and len(vals) > 5:
                    break
    return sum(vals) / len(vals) if vals else None

def find_baseline_step(target_loss, bl_steps, bl_losses):
    if target_loss >= bl_losses[0]:
        return 0.0
    # If target_loss is below baseline's final loss, extrapolate linearly
    # using the last segment's slope (baseline never reached this loss)
    if target_loss <= bl_losses[-1]:
        if len(bl_steps) >= 2 and bl_losses[-2] > bl_losses[-1]:
            slope = (bl_losses[-1] - bl_losses[-2]) / (bl_steps[-1] - bl_steps[-2])
            return bl_steps[-1] + (target_loss - bl_losses[-1]) / slope
        return float(bl_steps[-1])
    for i in range(len(bl_losses) - 1):
        if bl_losses[i] >= target_loss >= bl_losses[i + 1]:
            frac = (bl_losses[i] - target_loss) / (bl_losses[i] - bl_losses[i + 1])
            return bl_steps[i] + frac * (bl_steps[i + 1] - bl_steps[i])
    return float(bl_steps[-1])

# ── CLI ──────────────────────────────────────────────────

def pick_option(prompt, options):
    """Show numbered list and return chosen item."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    while True:
        try:
            choice = int(input("\n> "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
        except (ValueError, EOFError):
            pass
        print(f"  Escolha 1-{len(options)}")


def build_parser():
    parser = argparse.ArgumentParser(description="Interactive lag analysis for benchmark logs.")
    parser.add_argument(
        "--every",
        type=int,
        default=10,
        help="Show sampled rows every N steps. Default: 10.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a shorter report with fewer columns.",
    )
    return parser

def main():
    args = build_parser().parse_args()

    print("=" * 50)
    print("  LAG ANALYSIS — Experimento vs Baseline")
    print("=" * 50)

    # 1. Pick log with baseline
    logs = list_logs()
    if not logs:
        print(f"Nenhum log encontrado em {LOG_DIR}/")
        return

    log_names = [os.path.basename(f) for f in logs]
    bl_name = pick_option("Arquivo com o BASELINE:", log_names)
    bl_path = os.path.join(LOG_DIR, bl_name)

    # 2. Pick baseline section
    bl_sections = list_sections(bl_path)
    if not bl_sections:
        print("Nenhuma secao encontrada nesse log!")
        return
    bl_section = pick_option("Secao do BASELINE:", bl_sections)

    # 3. Pick log with experiment (can be same file)
    exp_name = pick_option("Arquivo com o EXPERIMENTO:", log_names)
    exp_path = os.path.join(LOG_DIR, exp_name)

    # 4. Pick experiment section
    exp_sections = list_sections(exp_path)
    if not exp_sections:
        print("Nenhuma secao encontrada nesse log!")
        return
    exp_section = pick_option("Secao do EXPERIMENTO:", exp_sections)

    # ── Parse & compute ──
    bl = parse_losses(bl_path, bl_section)
    exp = parse_losses(exp_path, exp_section)

    if not bl:
        print(f"Sem dados de loss para '{bl_section}'")
        return
    if not exp:
        print(f"Sem dados de loss para '{exp_section}'")
        return

    bl_steps = sorted(bl.keys())
    bl_losses = [bl[s] for s in bl_steps]
    exp_steps = sorted(exp.keys())

    # ── Print table ──
    short_exp = exp_section.split(":")[0].strip() if ":" in exp_section else exp_section
    print(f"\n{short_exp} vs {bl_section}")
    print("=" * 80)
    # Each bar char = 1% of loss difference relative to baseline at that step
    WINDOW = 20  # steps for windowed drop comparison
    sample_every = max(1, args.every)
    if args.summary:
        print(f"Sampling every {sample_every} steps (summary mode)")
        print(f"{'Step':>5} | {'EXP Loss':>9} | {'BL Loss':>9} | {'Diff%':>6} | {'Lag':>5}")
        print("-" * 47)
    else:
        print(f"Sampling every {sample_every} steps")
        print(f"{'Step':>5} | {'EXP Loss':>9} | {'BL Loss':>9} | {'Diff':>8} | {'Diff%':>6} | {'DrpΔ%':>6} | {'WinΔ%':>6} | {'Lag':>5} | Visual")
        print("-" * 98)

    # Initial losses for cumulative drop calculation
    exp_loss_0 = exp[exp_steps[0]]
    bl_loss_0 = bl[bl_steps[0]] if bl_steps else None

    all_diffs = []
    all_lags = []
    all_drop_deltas = []
    for s in exp_steps:
        el = exp[s]
        bl_at_s = bl.get(s)
        eq = find_baseline_step(el, bl_steps, bl_losses)
        lag = s - eq
        all_lags.append(lag)

        if bl_at_s is not None and bl_at_s != 0:
            diff = el - bl_at_s
            diff_pct = (diff / bl_at_s) * 100
        else:
            diff = None
            diff_pct = None
        all_diffs.append((s, diff, diff_pct))

        # Cumulative drop comparison: how much more % did EXP drop vs BL
        drop_delta_pct = None
        if bl_at_s is not None and bl_loss_0 is not None:
            bl_drop = bl_loss_0 - bl_at_s
            exp_drop = exp_loss_0 - el
            if bl_drop > 0:
                drop_delta_pct = (exp_drop - bl_drop) / bl_drop * 100
        all_drop_deltas.append((s, drop_delta_pct))

        # Windowed drop comparison: last WINDOW steps
        win_delta_pct = None
        si = exp_steps.index(s)
        if si >= WINDOW:
            s_back = exp_steps[si - WINDOW]
            exp_back = exp[s_back]
            bl_back = bl.get(s_back)
            if bl_at_s is not None and bl_back is not None:
                bl_win_drop = bl_back - bl_at_s
                exp_win_drop = exp_back - el
                if bl_win_drop > 0:
                    win_delta_pct = (exp_win_drop - bl_win_drop) / bl_win_drop * 100

        # Show periodic samples, plus first and last
        if s != exp_steps[0] and s != exp_steps[-1] and s % sample_every != 0:
            continue

        bl_str = f"{bl_at_s:>9.4f}" if bl_at_s is not None else f"{'—':>9}"
        diff_str = f"{diff:>+8.4f}" if diff is not None else f"{'—':>8}"
        pct_str = f"{diff_pct:>+6.1f}" if diff_pct is not None else f"{'—':>6}"
        drp_str = f"{drop_delta_pct:>+6.1f}" if drop_delta_pct is not None else f"{'—':>6}"
        win_str = f"{win_delta_pct:>+6.1f}" if win_delta_pct is not None else f"{'—':>6}"

        if args.summary:
            print(f"{s:>5} | {el:>9.4f} | {bl_str} | {pct_str} | {lag:>+5.1f}")
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

            print(f"{s:>5} | {el:>9.4f} | {bl_str} | {diff_str} | {pct_str} | {drp_str} | {win_str} | {lag:>+5.1f} | {bar}")

    print("-" * (47 if args.summary else 98))

    # ── Summary stats ──
    valid = [(s, d, p) for s, d, p in all_diffs if d is not None and s >= 10]
    if valid:
        last_s, last_d, last_p = valid[-1]
        best_d = min(valid, key=lambda x: x[1])
        worst_d = max(valid, key=lambda x: x[1])
        avg_d = sum(d for _, d, _ in valid) / len(valid)

        print(f"Diff final (step {last_s:>3}):  {last_d:+.4f}  ({last_p:+.1f}%)")
        print(f"Diff media (step 10+):  {avg_d:+.4f}")
        print(f"Melhor diff:            {best_d[1]:+.4f}  (step {best_d[0]})")
        print(f"Pior diff:              {worst_d[1]:+.4f}  (step {worst_d[0]})")

    # Drop comparison summary
    valid_drops = [(s, d) for s, d in all_drop_deltas if d is not None and s >= 10]
    if valid_drops:
        last_drop_s, last_drop_d = valid_drops[-1]
        print(f"Drop acum. EXP vs BL:   {last_drop_d:+.1f}%  (step {last_drop_s}) — (+)=EXP derrubou mais")

    lags_10 = [l for s, l in zip(exp_steps, all_lags) if s >= 10]
    if lags_10:
        print(f"Lag final:              {all_lags[-1]:+.1f} steps")

    # ── tok/sec comparison ──
    exp_tps = parse_avg_toksec(exp_path, exp_section)
    bl_tps = parse_avg_toksec(bl_path, bl_section)
    if exp_tps and bl_tps:
        pct = (exp_tps - bl_tps) / bl_tps * 100
        sign = "+" if pct >= 0 else ""
        print(f"\nThroughput: EXP {exp_tps:,.0f} tok/s vs BL {bl_tps:,.0f} tok/s ({sign}{pct:.1f}%)")

    print()
    print("(+) = EXP loss MAIOR que BL (pior)")
    print("(-) = EXP loss MENOR que BL (melhor)")

if __name__ == "__main__":
    main()
