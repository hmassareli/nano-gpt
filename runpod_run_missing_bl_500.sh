#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
else
  echo "Erro: .venv/bin/activate nao encontrado. Rode no Runpod dentro do repo ja configurado."
  exit 1
fi

STEPS="${1:-500}"
if [[ $# -gt 0 ]]; then
  shift
fi

DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
SEQ_LEN="${SEQ_LEN:-512}"

# Keep only the runs that still look decision-relevant at longer horizons.
# Excluded on purpose:
# - EXP-002: init-only fix already looks transient.
# - EXP-004: clearly harmful soft-tying variant.
RUNS=(
  "train.py"
  "experiments/train_exp003_conditioning_reg.py"
  "experiments/train_exp005_contrastive.py"
  "experiments/train_exp006_factored_head.py"
)

COMMON_ARGS=(
  "--steps=${STEPS}"
  "--quick-eval"
  "--device-batch-size=${DEVICE_BATCH_SIZE}"
  "--seq-len=${SEQ_LEN}"
)

if [[ $# -gt 0 ]]; then
  COMMON_ARGS+=("$@")
fi

echo "Running missing-baseline suite with ${#RUNS[@]} configs"
echo "steps=${STEPS} batch=${DEVICE_BATCH_SIZE} seq_len=${SEQ_LEN}"
echo "Configs: baseline + EXP-003 + EXP-005 + EXP-006"

python benchmark.py "${RUNS[@]}" "${COMMON_ARGS[@]}"
