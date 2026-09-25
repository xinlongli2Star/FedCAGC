#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_PATH="${DATA_PATH:-$PROJECT/data}"
GPU="${GPU:-0}"
mkdir -p "$PROJECT/results/experiments/tramark" "$PROJECT/logs/experiments/tramark"

# TraMark reuses the corresponding FedCAGC V2 partition/watermark split.
# Run scripts/run_main_3seeds.sh (at least the FedCAGC runs) first.
for DATASET in fmnist cifar10; do
  for SEED in 3047 3048 3049; do
    OUT="$PROJECT/results/experiments/tramark/${DATASET}_${SEED}"
    LOG="$PROJECT/logs/experiments/tramark/${DATASET}_${SEED}.log"
    echo "[RUN] TraMark dataset=$DATASET seed=$SEED"
    CUDA_VISIBLE_DEVICES="$GPU" python -u "$PROJECT/fedcagc_v2_tramark.py" \
      --project "$PROJECT" --data_path "$DATA_PATH" --dataset "$DATASET" --seed "$SEED" --device cuda \
      --num_clients 10 --rounds 100 --local_epochs 5 --local_bs 64 --local_lr 0.01 \
      --momentum 0.9 --weight_decay 1e-4 --gamma 0.5 \
      --wm_train_size 100 --wm_test_size 200 \
      --alpha 0.5 --k 0.01 --wm_epochs 5 --wm_lr 0.0001 --wm_momentum 0 --wm_bs 32 \
      --wm_grad_mode official_accumulate --wm_transform official \
      --eval_every 5 --eval_tail 10 \
      --output_dir "$OUT" 2>&1 | tee "$LOG"
  done
done
