#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
GPU="${GPU:-0}"

mkdir -p "$PROJECT/results/experiments/robustness" "$PROJECT/logs/experiments/robustness"

for DATASET in fmnist cifar10; do
  for SEED in 3047 3048 3049; do
    OUT="$PROJECT/results/experiments/robustness/${DATASET}_${SEED}"
    LOG="$PROJECT/logs/experiments/robustness/${DATASET}_${SEED}.log"
    CUDA_VISIBLE_DEVICES="$GPU" python -u "$PROJECT/fedcagc_v2_robustness.py" \
      --project "$PROJECT" --dataset "$DATASET" --seed "$SEED" --device cuda \
      --batch_size 64 --ft_epochs 50 --ft_lr 0.01 \
      --momentum 0.9 --weight_decay 1e-4 --grad_clip_norm 20 \
      --prune_ratios 0,10,20,40,60,80,90,95,99 \
      --output_dir "$OUT" 2>&1 | tee "$LOG"
  done
done
