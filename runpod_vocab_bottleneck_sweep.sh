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
ANALYSIS_STEPS="${ANALYSIS_STEPS:-0}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
SEQ_LEN="${SEQ_LEN:-512}"
NUM_SHARDS="${NUM_SHARDS:-10}"
VOCABS=("${@:2}")

if [[ ${#VOCABS[@]} -eq 0 ]]; then
  VOCABS=(4096 8192 16384 32768)
fi

echo "Running vocab bottleneck sweep"
echo "train_steps=${STEPS} analysis_steps=${ANALYSIS_STEPS} batch=${DEVICE_BATCH_SIZE} eval_batch=${EVAL_BATCH_SIZE} seq_len=${SEQ_LEN}"
echo "vocabs=${VOCABS[*]}"

for vocab in "${VOCABS[@]}"; do
  tokenizer_dir="${HOME}/.cache/autoresearch/tokenizer_v${vocab}"

  echo
  echo "===================================================================="
  echo "Preparing tokenizer vocab=${vocab} -> ${tokenizer_dir}"
  echo "===================================================================="
  python prepare.py \
    --num-shards="${NUM_SHARDS}" \
    --vocab-size="${vocab}" \
    --tokenizer-dir="${tokenizer_dir}"

  echo
  echo "===================================================================="
  echo "Gradient bottleneck analysis for vocab=${vocab}"
  echo "===================================================================="
  AUTORESEARCH_TOKENIZER_DIR="${tokenizer_dir}" \
    python gradient_bottleneck_analysis.py --baseline --steps="${ANALYSIS_STEPS}" --tokenizer-dir="${tokenizer_dir}"

  echo
  echo "===================================================================="
  echo "Baseline benchmark for vocab=${vocab}"
  echo "===================================================================="
  AUTORESEARCH_TOKENIZER_DIR="${tokenizer_dir}" \
    python benchmark.py train.py \
      --steps="${STEPS}" \
      --quick-eval \
      --device-batch-size="${DEVICE_BATCH_SIZE}" \
      --eval-batch-size="${EVAL_BATCH_SIZE}" \
      --seq-len="${SEQ_LEN}"
done
