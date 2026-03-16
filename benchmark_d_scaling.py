"""Benchmark sweep for baseline width D (n_embd).

Compares multiple baseline widths using the same training script and prints
token-budget and wall-clock convergence summaries.
"""

import os
import re
import subprocess
import sys
import time
from datetime import datetime
from statistics import fmean


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "benchmark_logs")
os.makedirs(LOGS_DIR, exist_ok=True)

STEP_RE = re.compile(r"step\s+(\d+).*?loss:\s+([\d.]+).*?dt:\s+([\d.]+)ms.*?tok/sec:\s+([\d,]+)")
BPB_RE = re.compile(r"val_bpb:\s+([\d.]+)")
TOKENS_RE = re.compile(r"total_tokens_M:\s+([\d.]+)")
VRAM_RE = re.compile(r"peak_vram_mb:\s+([\d.]+)")
TIME_RE = re.compile(r"training_seconds:\s+([\d.]+)")
TOTAL_TIME_RE = re.compile(r"total_seconds:\s+([\d.]+)")


def get_python_executable():
    candidates = [
        os.path.join(SCRIPT_DIR, ".venv", "Scripts", "python.exe"),
        os.path.join(SCRIPT_DIR, ".venv", "bin", "python"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return sys.executable


def parse_args(argv):
    opts = {
        "dims": [384, 512, 640, 768],
        "steps": 100,
        "time_budget": 0,
        "device_batch_size": 16,
        "seq_len": 512,
        "eval_mode": "quick",
        "use_compile": False,
        "cooldown": 0,
    }
    for arg in argv:
        if arg.startswith("--dims="):
            dims = [int(part) for part in arg.split("=", 1)[1].split(",") if part.strip()]
            opts["dims"] = dims
        elif arg.startswith("--steps="):
            opts["steps"] = int(arg.split("=", 1)[1])
        elif arg.startswith("--time-budget="):
            opts["time_budget"] = int(arg.split("=", 1)[1])
        elif arg.startswith("--device-batch-size="):
            opts["device_batch_size"] = int(arg.split("=", 1)[1])
        elif arg.startswith("--seq-len="):
            opts["seq_len"] = int(arg.split("=", 1)[1])
        elif arg.startswith("--cooldown="):
            opts["cooldown"] = int(arg.split("=", 1)[1])
        elif arg == "--full-eval":
            opts["eval_mode"] = "full"
        elif arg == "--quick-eval":
            opts["eval_mode"] = "quick"
        elif arg == "--no-eval":
            opts["eval_mode"] = "none"
        elif arg == "--compile":
            opts["use_compile"] = True
        elif arg == "--no-compile":
            opts["use_compile"] = False
        else:
            raise SystemExit(f"Unknown arg: {arg}")
    if opts["time_budget"] and opts["steps"]:
        opts["steps"] = 0
    return opts


def build_log_path(dims, steps, time_budget):
    stamp = datetime.now().strftime("%m%d_%H%M")
    dims_tag = "_".join(f"d{dim}" for dim in dims)
    budget_tag = f"{steps}steps" if steps else f"{time_budget}s"
    filename = f"bench_dsweep_{dims_tag}_{stamp}_{budget_tag}.log"
    return os.path.join(LOGS_DIR, filename)


def log_factory(log_file):
    def log(msg="", end="\n"):
        print(msg, end=end, flush=True)
        log_file.write(msg + end)
        log_file.flush()
    return log


def parse_step_data(lines):
    data = []
    cumulative_s = 0.0
    for line in lines:
        match = STEP_RE.search(line)
        if not match:
            continue
        dt_s = float(match.group(3)) / 1000.0
        cumulative_s += dt_s
        data.append({
            "step": int(match.group(1)),
            "loss": float(match.group(2)),
            "dt_s": dt_s,
            "cum_s": cumulative_s,
            "tok_sec": int(match.group(4).replace(",", "")),
        })
    return data


def value_at_or_before_step(step_data, checkpoint):
    last = None
    for row in step_data:
        if row["step"] > checkpoint:
            break
        last = row
    return last["loss"] if last else None


def value_at_or_before_time(step_data, checkpoint_s):
    last = None
    for row in step_data:
        if row["cum_s"] > checkpoint_s:
            break
        last = row
    return last["loss"] if last else None


def choose_time_checkpoints(common_runtime_s):
    candidates = [30, 60, 120, 300, 600, 900, 1200, 1800, 3600]
    return [value for value in candidates if value <= common_runtime_s]


def format_value(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def print_summary(log, results, opts):
    log("\n" + "=" * 88)
    mode = f"{opts['steps']} steps" if opts["steps"] else f"{opts['time_budget']}s"
    log(f"  D SWEEP SUMMARY ({mode})")
    log("=" * 88)
    header = (
        f"{'D':>5} {'val_bpb':>10} {'loss_f':>9} {'drop':>9} {'drop/min':>10} "
        f"{'avg_tok/s':>10} {'train_s':>9} {'VRAM_MB':>8} {'exit':>5}"
    )
    log(header)
    log("-" * len(header))
    for result in results:
        step_data = result["step_data"]
        loss_i = step_data[0]["loss"] if step_data else None
        loss_f = step_data[-1]["loss"] if step_data else None
        train_s = result["time_s"] or (step_data[-1]["cum_s"] if step_data else None)
        drop = (loss_i - loss_f) if loss_i is not None and loss_f is not None else None
        drop_per_min = (drop / (train_s / 60.0)) if drop is not None and train_s and train_s > 0 else None
        avg_toksec = fmean(row["tok_sec"] for row in step_data) if step_data else None
        log(
            f"{result['dim']:>5} {format_value(result['val_bpb'], 6):>10} {format_value(loss_f, 4):>9} "
            f"{format_value(drop, 4):>9} {format_value(drop_per_min, 4):>10} {format_value(avg_toksec, 0):>10} "
            f"{format_value(train_s, 1):>9} {format_value(result['vram_mb'], 0):>8} {result['exit_code']:>5}"
        )

    valid = [result for result in results if result["step_data"]]
    if not valid:
        return

    min_last_step = min(result["step_data"][-1]["step"] for result in valid)
    step_checkpoints = [cp for cp in [10, 20, 50, 100, 200, 500, 1000] if cp <= min_last_step]
    if step_checkpoints:
        log("\nLoss at common steps:")
        header = "checkpoint".ljust(10) + " " + " ".join(f"D={result['dim']:<10}" for result in valid)
        log(header)
        log("-" * len(header))
        for checkpoint in step_checkpoints:
            row = [f"{checkpoint:<10}"]
            for result in valid:
                row.append(f"{format_value(value_at_or_before_step(result['step_data'], checkpoint), 4):<10}")
            log(" ".join(row))

    common_runtime_s = min(result["step_data"][-1]["cum_s"] for result in valid)
    time_checkpoints = choose_time_checkpoints(common_runtime_s)
    if time_checkpoints:
        log("\nLoss at common wall-clock checkpoints:")
        header = "checkpoint".ljust(10) + " " + " ".join(f"D={result['dim']:<10}" for result in valid)
        log(header)
        log("-" * len(header))
        for checkpoint in time_checkpoints:
            label = f"{checkpoint // 60}m" if checkpoint % 60 == 0 and checkpoint >= 60 else f"{checkpoint}s"
            row = [f"{label:<10}"]
            for result in valid:
                row.append(f"{format_value(value_at_or_before_time(result['step_data'], checkpoint), 4):<10}")
            log(" ".join(row))


def main():
    opts = parse_args(sys.argv[1:])
    python_exe = get_python_executable()
    log_path = build_log_path(opts["dims"], opts["steps"], opts["time_budget"])

    with open(log_path, "w", encoding="utf-8") as log_file:
        log = log_factory(log_file)
        log("=" * 72)
        log(f"  D SWEEP: dims={opts['dims']}")
        budget_label = f"steps={opts['steps']}" if opts["steps"] else f"time_budget={opts['time_budget']}s"
        log(f"  {budget_label} | seq_len={opts['seq_len']} | device_batch_size={opts['device_batch_size']}")
        log(f"  eval={opts['eval_mode']} | compile={'on' if opts['use_compile'] else 'off'}")
        log(f"  Log: {log_path}")
        log("=" * 72)

        results = []
        for index, dim in enumerate(opts["dims"], start=1):
            label = f"Baseline D={dim}"
            log("\n" + "-" * 72)
            log(f"  [{index}/{len(opts['dims'])}] {label}")
            log("-" * 72)

            cmd = [
                python_exe,
                os.path.join(SCRIPT_DIR, "train.py"),
                f"--n-embd={dim}",
                f"--device-batch-size={opts['device_batch_size']}",
                f"--seq-len={opts['seq_len']}",
            ]
            if opts["steps"]:
                cmd.append(f"--steps={opts['steps']}")
            if not opts["use_compile"]:
                cmd.append("--no-compile")
            if opts["eval_mode"] == "quick":
                cmd.append("--quick-eval")
            elif opts["eval_mode"] == "none":
                cmd.append("--no-eval")

            env = os.environ.copy()
            if opts["time_budget"]:
                env["NANOGPT_TIME_BUDGET"] = str(opts["time_budget"])

            start = time.time()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )

            output_lines = []
            for line in proc.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
                output_lines.append(line)
            proc.wait()
            wall_s = time.time() - start
            output = "".join(output_lines)

            results.append({
                "dim": dim,
                "label": label,
                "val_bpb": float(BPB_RE.search(output).group(1)) if BPB_RE.search(output) else None,
                "tokens_M": float(TOKENS_RE.search(output).group(1)) if TOKENS_RE.search(output) else None,
                "vram_mb": float(VRAM_RE.search(output).group(1)) if VRAM_RE.search(output) else None,
                "time_s": float(TIME_RE.search(output).group(1)) if TIME_RE.search(output) else None,
                "total_s": float(TOTAL_TIME_RE.search(output).group(1)) if TOTAL_TIME_RE.search(output) else None,
                "wall_s": wall_s,
                "exit_code": proc.returncode,
                "step_data": parse_step_data(output_lines),
            })

            if proc.returncode != 0:
                log(f"  *** FAILED (exit code {proc.returncode}) ***")

            if opts["cooldown"] > 0 and index < len(opts["dims"]):
                log(f"\n  Cooldown: {opts['cooldown']}s")
                time.sleep(opts["cooldown"])

        print_summary(log, results, opts)


if __name__ == "__main__":
    main()