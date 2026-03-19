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
SEQ_LEN="${SEQ_LEN:-512}"
NUM_SHARDS="${NUM_SHARDS:-10}"

tokenizer_dir="${HOME}/.cache/autoresearch/tokenizer_v${VOCAB_SIZE}"

echo "Benchmark multihead com vocab grande"
echo "steps=${STEPS} vocab=${VOCAB_SIZE} batch=${DEVICE_BATCH_SIZE} seq_len=${SEQ_LEN}"
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
echo "Benchmark baseline + shortlist multihead"
echo "===================================================================="
python benchmark.py \
  train.py \
  experiments/train_exp007_multi_head.py \
  experiments/train_exp009_multi_head_two_stage.py \
  experiments/train_exp017_2head_per_head_ce.py \
  experiments/train_exp018_4head_per_head_ce.py \
  --steps="${STEPS}" \
  --quick-eval \
  --device-batch-size="${DEVICE_BATCH_SIZE}" \
  --seq-len="${SEQ_LEN}"