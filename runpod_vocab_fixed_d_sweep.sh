#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
else
  echo "Erro: .venv/bin/activate nao encontrado. Rode no Runpod dentro do repo ja configurado."
  exit 1
fi

STEPS="${1:-300}"
VOCAB_SIZE="${VOCAB_SIZE:-32768}"
ANALYSIS_STEPS="${ANALYSIS_STEPS:-0}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
SEQ_LEN="${SEQ_LEN:-512}"
NUM_SHARDS="${NUM_SHARDS:-10}"
D_VALUES=("${@:2}")

if [[ ${#D_VALUES[@]} -eq 0 ]]; then
  D_VALUES=(256 384 512 768 1024)
fi

tokenizer_dir="${HOME}/.cache/autoresearch/tokenizer_v${VOCAB_SIZE}"

echo "Running fixed-V, D-sweep"
echo "vocab=${VOCAB_SIZE} train_steps=${STEPS} analysis_steps=${ANALYSIS_STEPS} batch=${DEVICE_BATCH_SIZE} seq_len=${SEQ_LEN}"
echo "D values=${D_VALUES[*]}"

echo
echo "===================================================================="
echo "Preparing tokenizer vocab=${VOCAB_SIZE} -> ${tokenizer_dir}"
echo "===================================================================="
python prepare.py \
  --num-shards="${NUM_SHARDS}" \
  --vocab-size="${VOCAB_SIZE}" \
  --tokenizer-dir="${tokenizer_dir}"

for d_model in "${D_VALUES[@]}"; do
  echo
  echo "===================================================================="
  echo "Gradient bottleneck analysis for vocab=${VOCAB_SIZE}, D=${d_model}"
  echo "===================================================================="
  AUTORESEARCH_TOKENIZER_DIR="${tokenizer_dir}" \
    python gradient_bottleneck_analysis.py \
      --baseline \
      --steps="${ANALYSIS_STEPS}" \
      --tokenizer-dir="${tokenizer_dir}" \
      --n-embd="${d_model}"

  echo
  echo "===================================================================="
  echo "Baseline benchmark for vocab=${VOCAB_SIZE}, D=${d_model}"
  echo "===================================================================="
  AUTORESEARCH_TOKENIZER_DIR="${tokenizer_dir}" \
    python benchmark.py train.py \
      --steps="${STEPS}" \
      --quick-eval \
      --device-batch-size="${DEVICE_BATCH_SIZE}" \
      --seq-len="${SEQ_LEN}" \
      --n-embd="${d_model}"
done
