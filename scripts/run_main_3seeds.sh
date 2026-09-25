#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_PATH="${DATA_PATH:-$PROJECT/data}"
GPU="${GPU:-0}"
mkdir -p "$DATA_PATH" "$PROJECT/results/main" "$PROJECT/logs/main"

for DATASET in fmnist cifar10; do
  for METHOD in fedavg fedipr flwb fedawm fedcagc; do
    for SEED in 3047 3048 3049; do
      OUT="$PROJECT/results/main/${DATASET}_${METHOD}_${SEED}"
      LOG="$PROJECT/logs/main/${DATASET}_${METHOD}_${SEED}.log"
      echo "[RUN] dataset=$DATASET method=$METHOD seed=$SEED"
      CUDA_VISIBLE_DEVICES="$GPU" python -u "$PROJECT/fedcagc_v2_all_methods.py" \
        --dataset "$DATASET" --method "$METHOD" \
        --data_path "$DATA_PATH" --device cuda \
        --num_clients 10 --rounds 100 --seed "$SEED" \
        --local_epochs 5 --local_bs 64 --local_lr 0.01 \
        --momentum 0.9 --weight_decay 1e-4 \
        --non_iid --gamma 0.5 \
        --watermark_source mnist --wm_train_size 100 --wm_test_size 200 --wm_beta 1.0 \
        --surgery_scope final_classifier --ema_rho 0.9 --tau_neg 0.0 \
        --wm_compensate --max_comp_scale 2.0 --grad_clip_norm 20 \
        --eval_every 5 --eval_tail 10 --snapshot_rounds 1,10,20,50,100 \
        --output_dir "$OUT" 2>&1 | tee "$LOG"
    done
  done
done
