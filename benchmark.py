"""
Benchmark: compara diferentes experimentos de treino.
Uso:
    uv run benchmark.py train.py                           # roda 1 arquivo, 50 steps
    uv run benchmark.py train.py train_exp002.py           # compara 2 arquivos
    uv run benchmark.py train.py train_exp002.py --steps=100  # mais steps
    uv run benchmark.py experiments/train_*.py --no-eval   # glob de experimentos
    uv run benchmark.py --steps=100 --full-eval             # sem args = roda train.py
"""

import subprocess
import sys
import re
import os
import time
import glob
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "benchmark_logs")
os.makedirs(LOGS_DIR, exist_ok=True)

def get_python_executable():
    candidates = [
        os.path.join(SCRIPT_DIR, ".venv", "Scripts", "python.exe"),
        os.path.join(SCRIPT_DIR, ".venv", "bin", "python"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return sys.executable

VENV_PYTHON = get_python_executable()

STEPS = 50
EVAL_MODE = "quick"
COOLDOWN_SECONDS = 0
train_files = []
train_script_args = []

for arg in sys.argv[1:]:
    if arg.startswith("--steps="):
        STEPS = int(arg.split("=", 1)[1])
    elif arg.startswith("--cooldown="):
        COOLDOWN_SECONDS = int(arg.split("=", 1)[1])
    elif arg == "--full-eval":
        EVAL_MODE = "full"
    elif arg == "--no-eval":
        EVAL_MODE = "none"
    elif arg == "--quick-eval":
        EVAL_MODE = "quick"
    elif arg.startswith("--"):
        train_script_args.append(arg)
    else:
        # Treat as train file (support glob)
        expanded = glob.glob(arg)
        if expanded:
            train_files.extend(expanded)
        elif os.path.isfile(arg):
            train_files.append(arg)
        else:
            print(f"Warning: '{arg}' not found, skipping")

# Default: just train.py
if not train_files:
    train_files = ["train.py"]

# Extract EXP_TITLE from a train script file
TITLE_RE = re.compile(r'^EXP_TITLE\s*=\s*["\'](.+?)["\']', re.MULTILINE)
EXP_ID_RE = re.compile(r'\bEXP-(\d{3}(?:\.\d+)*)\b', re.IGNORECASE)
def get_exp_title(filepath):
    with open(filepath, encoding="utf-8") as fh:
        head = fh.read(2048)
    m = TITLE_RE.search(head)
    return m.group(1) if m else os.path.splitext(os.path.basename(filepath))[0]

def split_exp_id(exp_id):
    return [int(part) for part in exp_id.split(".")]

def get_config_tag(name, title):
    for candidate in (title, name):
        m = EXP_ID_RE.search(candidate)
        if m:
            return "exp" + m.group(1).replace(".", "-")
    lowered = title.lower()
    if lowered == "baseline" or name == "train":
        return "bl"
    tag = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return tag[:24] or "run"

def sort_config_tags(tags):
    def sort_key(tag):
        if tag.startswith("exp"):
            parts = tag[3:].split("-")
            if parts and all(part.isdigit() for part in parts):
                return (0, tuple(int(part) for part in parts), tag)
        if tag == "bl":
            return (2, (0,), tag)
        return (1, (0,), tag)
    return sorted(tags, key=sort_key)

def build_log_path(configs, steps):
    tags = []
    seen = set()
    for cfg in configs:
        tag = get_config_tag(cfg["name"], cfg["title"])
        if tag not in seen:
            tags.append(tag)
            seen.add(tag)
    tags = sort_config_tags(tags)
    stamp = datetime.now().strftime("%m%d_%H%M")
    filename = f"bench_{'_'.join(tags)}_{stamp}_{steps}steps.log"
    path = os.path.join(LOGS_DIR, filename)
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(filename)
    suffix = 2
    while True:
        candidate = os.path.join(LOGS_DIR, f"{stem}_{suffix}{ext}")
        if not os.path.exists(candidate):
            return candidate
        suffix += 1

# Build configs from file list
CONFIGS = []
for f in train_files:
    name = os.path.splitext(os.path.basename(f))[0]
    title = get_exp_title(f)
    CONFIGS.append({"name": name, "title": title, "script": f})

BPB_RE = re.compile(r"val_bpb:\s+([\d.]+)")
TOKENS_RE = re.compile(r"total_tokens_M:\s+([\d.]+)")
STEPS_RE = re.compile(r"num_steps:\s+(\d+)")
VRAM_RE = re.compile(r"peak_vram_mb:\s+([\d.]+)")
TIME_RE = re.compile(r"training_seconds:\s+([\d.]+)")
TOTAL_TIME_RE = re.compile(r"total_seconds:\s+([\d.]+)")
STEP_RE = re.compile(r"step (\d+).*?loss:\s+([\d.]+).*?dt:\s+([\d.]+)ms.*?tok/sec:\s+([\d,]+)")

results = []

eval_label = {"quick": "quick-eval", "full": "full-eval", "none": "sem eval"}[EVAL_MODE]

# Log file com timestamp
log_path = build_log_path(CONFIGS, STEPS)
log_file = open(log_path, "w", encoding="utf-8")

def log(msg="", end="\n"):
    """Print to terminal and write to log file."""
    print(msg, end=end, flush=True)
    log_file.write(msg + end)
    log_file.flush()

log(f"\n{'='*60}")
log(f"  BENCHMARK: {len(CONFIGS)} configs x {STEPS} steps  ({eval_label})")
log(f"  Log: {log_path}")
log(f"{'='*60}")

bench_start = time.time()

for i, cfg in enumerate(CONFIGS):
    label = cfg["title"]
    remaining = len(CONFIGS) - i
    if results:
        avg_time = (time.time() - bench_start) / len(results)
        eta_min = avg_time * remaining / 60
        eta_str = f"  ~{eta_min:.0f}min restantes" if eta_min >= 1 else f"  ~{eta_min*60:.0f}s restantes"
    else:
        eta_str = ""
    log(f"\n{'-'*60}")
    log(f"  [{i+1}/{len(CONFIGS)}] {label}  ({STEPS} steps){eta_str}")
    log(f"{'-'*60}\n")

    cmd = [
        VENV_PYTHON, cfg["script"],
        f"--steps={STEPS}",
        "--seq-len=512",
        "--device-batch-size=16",
        "--no-compile",
    ]
    cmd.extend(train_script_args)
    if EVAL_MODE == "none":
        cmd.append("--no-eval")
    elif EVAL_MODE == "quick":
        cmd.append("--quick-eval")
    # full: don't add any eval flag (default full eval)

    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output_lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        log_file.write(line)
        log_file.flush()
        output_lines.append(line)
    proc.wait()
    wall_time = time.time() - t0
    output = "".join(output_lines)

    bpb = BPB_RE.search(output)
    tokens = TOKENS_RE.search(output)
    steps = STEPS_RE.search(output)
    vram = VRAM_RE.search(output)
    train_time = TIME_RE.search(output)
    total_time_m = TOTAL_TIME_RE.search(output)

    results.append({
        "name": label,
        "val_bpb": float(bpb.group(1)) if bpb else None,
        "tokens_M": float(tokens.group(1)) if tokens else None,
        "steps": int(steps.group(1)) if steps else None,
        "vram_mb": float(vram.group(1)) if vram else None,
        "time_s": float(train_time.group(1)) if train_time else None,
        "wall_s": wall_time,
        "exit_code": proc.returncode,
    })

    # Parse per-step loss curve
    step_data = []
    for line in output_lines:
        m = STEP_RE.search(line)
        if m:
            step_data.append({
                "step": int(m.group(1)),
                "loss": float(m.group(2)),
                "dt_ms": float(m.group(3)),
                "tok_sec": int(m.group(4).replace(",", "")),
            })
    results[-1]["step_data"] = step_data

    if proc.returncode != 0:
        log(f"  *** FALHOU (exit code {proc.returncode}) ***")

    if i < len(CONFIGS) - 1 and COOLDOWN_SECONDS > 0:
        log(f"\n  Cooldown: {COOLDOWN_SECONDS}s para GPU esfriar...")
        time.sleep(COOLDOWN_SECONDS)

# Summary table
total_bench_time = time.time() - bench_start
log(f"\n\n{'='*80}")
log(f"  RESULTADOS  ({STEPS} steps, {eval_label}, total: {total_bench_time/60:.1f} min)")
log(f"{'='*80}")
if EVAL_MODE != "none":
    header = f"{'Config':<42} {'val_bpb':>10} {'tokens_M':>10} {'VRAM_MB':>8} {'treino_s':>9} {'total_s':>8} {'exit':>5}"
else:
    header = f"{'Config':<42} {'tokens_M':>10} {'VRAM_MB':>8} {'treino_s':>9} {'total_s':>8} {'exit':>5}"
log(header)
log("-" * len(header))
for r in results:
    bpb_str = f"{r['val_bpb']:.6f}" if r['val_bpb'] is not None else "-"
    tok_str = f"{r['tokens_M']:.1f}" if r['tokens_M'] is not None else "-"
    vram_str = f"{r['vram_mb']:.0f}" if r['vram_mb'] is not None else "-"
    time_str = f"{r['time_s']:.1f}" if r['time_s'] is not None else "-"
    wall_str = f"{r['wall_s']:.0f}"
    if EVAL_MODE != "none":
        log(f"{r['name']:<42} {bpb_str:>10} {tok_str:>10} {vram_str:>8} {time_str:>9} {wall_str:>8} {r['exit_code']:>5}")
    else:
        log(f"{r['name']:<42} {tok_str:>10} {vram_str:>8} {time_str:>9} {wall_str:>8} {r['exit_code']:>5}")

# Best result
valid = [r for r in results if r['val_bpb'] is not None]
if valid:
    best = min(valid, key=lambda r: r['val_bpb'])
    worst = max(valid, key=lambda r: r['val_bpb'])
    delta = worst['val_bpb'] - best['val_bpb']
    log(f"\nMelhor:  {best['name']}  val_bpb = {best['val_bpb']:.6f}")
    log(f"Pior:    {worst['name']}  val_bpb = {worst['val_bpb']:.6f}")
    log(f"Delta:   {delta:.6f} bpb")
    if delta < 0.005:
        log("(delta pequeno — considere rodar com mais steps para confirmar)")
elif EVAL_MODE == "none":
    fastest = min(results, key=lambda r: r['time_s'] or float('inf'))
    log(f"\nMais rapido: {fastest['name']}  treino={fastest['time_s']:.1f}s")

# Convergence metrics
complete = [r for r in results if r.get('step_data') and len(r['step_data']) >= 2 and r['time_s']]
if complete:
    log(f"\n{'='*80}")
    log(f"  METRICAS DE CONVERGENCIA")
    log(f"{'='*80}")
    header2 = f"{'Config':<42} {'loss_i':>8} {'loss_f':>8} {'drop':>8} {'drop/min':>10} {'avg_tok/s':>10} {'loss/Mtok':>10}"
    log(header2)
    log("-" * len(header2))
    for r in complete:
        sd = r['step_data']
        loss_initial = sd[0]['loss']
        loss_final = sd[-1]['loss']
        loss_drop = loss_initial - loss_final
        train_min = r['time_s'] / 60
        drop_per_min = loss_drop / train_min if train_min > 0 else 0
        avg_tok_sec = sum(s['tok_sec'] for s in sd) / len(sd)
        tokens_M = r['tokens_M'] or 0
        loss_per_Mtok = loss_drop / tokens_M if tokens_M > 0 else 0
        log(f"{r['name']:<42} {loss_initial:>8.3f} {loss_final:>8.3f} {loss_drop:>8.3f} {drop_per_min:>10.4f} {avg_tok_sec:>10.0f} {loss_per_Mtok:>10.4f}")

    # Loss at common milestones
    milestones = [5, 10, 25, 50, 100]
    active_milestones = [m for m in milestones if m <= STEPS]
    if len(complete) > 1 and active_milestones:
        log(f"\n  Loss por step (comparacao direta):")
        ms_header = f"  {'Step':>6}" + "".join(f" {r['name']:>30}" for r in complete)
        log(ms_header)
        log("  " + "-" * (len(ms_header) - 2))
        for ms in active_milestones:
            row = f"  {ms:>6}"
            for r in complete:
                sd = r['step_data']
                match = [s for s in sd if s['step'] == ms]
                if match:
                    row += f" {match[0]['loss']:>30.4f}"
                else:
                    row += f" {'—':>30}"
            log(row)

    # Elapsed time at milestones
    if len(complete) > 1 and active_milestones:
        log(f"\n  Tempo acumulado por step (s):")
        ms_header = f"  {'Step':>6}" + "".join(f" {r['name']:>30}" for r in complete)
        log(ms_header)
        log("  " + "-" * (len(ms_header) - 2))
        for ms in active_milestones:
            row = f"  {ms:>6}"
            for r in complete:
                sd = r['step_data']
                steps_up_to = [s for s in sd if s['step'] <= ms]
                if steps_up_to:
                    elapsed = sum(s['dt_ms'] for s in steps_up_to) / 1000
                    row += f" {elapsed:>30.1f}"
                else:
                    row += f" {'—':>30}"
            log(row)

log_file.close()
print(f"\nLog salvo em: {log_path}")
