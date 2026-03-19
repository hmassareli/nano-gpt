#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
else
  echo "Erro: .venv/bin/activate nao encontrado. Rode no Runpod dentro do repo ja configurado."
  exit 1
fi

STEPS="${1:-1000}"
VOCAB_SIZE="${VOCAB_SIZE:-32768}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
SEQ_LEN="${SEQ_LEN:-512}"
NUM_SHARDS="${NUM_SHARDS:-10}"

tokenizer_dir="${HOME}/.cache/autoresearch/tokenizer_v${VOCAB_SIZE}"

echo "Benchmark EXP-029 com vocab grande"
echo "steps=${STEPS} vocab=${VOCAB_SIZE} batch=${DEVICE_BATCH_SIZE} eval_batch=${EVAL_BATCH_SIZE} seq_len=${SEQ_LEN}"
echo "tokenizer_dir=${tokenizer_dir}"

echo
echo "===================================================================="
echo "Preparing tokenizer vocab=${VOCAB_SIZE} -> ${tokenizer_dir}"
echo "===================================================================="
python prepare.py \
  --num-shards="${NUM_SHARDS}" \
  --vocab-size="${VOCAB_SIZE}" \
  --tokenizer-dir="${tokenizer_dir}"

export AUTORESEARCH_TOKENIZER_DIR="${tokenizer_dir}"

echo
echo "===================================================================="
echo "Benchmark baseline + EXP-029.x"
echo "===================================================================="
python benchmark.py \
  train.py \
  experiments/train_exp029_1_jepa_latent.py \
  experiments/train_exp029_2_next_token_latent.py \
  experiments/train_exp029_3_ema_teacher.py \
  --steps="${STEPS}" \
  --quick-eval \
  --device-batch-size="${DEVICE_BATCH_SIZE}" \
  --eval-batch-size="${EVAL_BATCH_SIZE}" \
  --seq-len="${SEQ_LEN}"