#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
else
  echo "Erro: .venv/bin/activate nao encontrado. Rode no pod dentro do repo ja configurado."
  exit 1
fi

STEPS="${1:-500}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
SEQ_LEN="${SEQ_LEN:-512}"

COMMON_ARGS=(
  "--steps=${STEPS}"
  "--quick-eval"
  "--device-batch-size=${DEVICE_BATCH_SIZE}"
  "--seq-len=${SEQ_LEN}"
)

# Selected suite:
# - include baseline and EXP-007 as requested
# - include EXP-012..025 except the clearly bad runs seen so far:
#   EXP-018, EXP-019, EXP-022
RUNS=(
  "train.py"
  "experiments/train_exp007_multi_head.py"
  "experiments/train_exp012_2head_full_rank.py"
  "experiments/train_exp013_4head_full_rank.py"
  "experiments/train_exp014_cosine_diversity.py"
  "experiments/train_exp015_decov.py"
  "experiments/train_exp016_ortho_reg.py"
  "experiments/train_exp017_2head_per_head_ce.py"
  "experiments/train_exp020_4head_token_gating.py"
  "experiments/train_exp021_deep_supervision.py"
  "experiments/train_exp023_mixture_of_softmax.py"
  "experiments/train_exp024_temp_schedule.py"
  "experiments/train_exp025_4head_grad_diversity.py"
)

echo "Running ${#RUNS[@]} configs with steps=${STEPS}, batch=${DEVICE_BATCH_SIZE}, seq_len=${SEQ_LEN}"

for script in "${RUNS[@]}"; do
  echo
  echo "============================================================"
  echo "Running ${script}"
  echo "============================================================"
  python benchmark.py "$script" "${COMMON_ARGS[@]}"
done
