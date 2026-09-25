#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_PATH="${DATA_PATH:-$PROJECT/data}"
GPU="${GPU:-0}"

mkdir -p "$PROJECT/results/experiments/mechanism" "$PROJECT/logs/experiments/mechanism"

for SEED in 3047 3048 3049; do
  COMMON=(
    --dataset cifar10 --data_path "$DATA_PATH" --device cuda
    --num_clients 10 --rounds 100 --seed "$SEED"
    --local_epochs 5 --local_bs 64 --local_lr 0.01
    --momentum 0.9 --weight_decay 1e-4
    --non_iid --gamma 0.5
    --watermark_source mnist --wm_train_size 100 --wm_test_size 200 --wm_beta 1.0
    --ema_rho 0.9 --tau_neg 0.0 --wm_compensate --max_comp_scale 2.0
    --grad_clip_norm 20 --eval_every 5 --eval_tail 10
    --snapshot_rounds 1,10,20,50,100
  )

  OUT="$PROJECT/results/experiments/mechanism/fresh_round_final_${SEED}"
  LOG="$PROJECT/logs/experiments/mechanism/fresh_round_final_${SEED}.log"
  CUDA_VISIBLE_DEVICES="$GPU" python -u "$PROJECT/fedcagc_v2_mechanism_experiments.py" \
    --method fedcagc --prototype_mode fresh_round_start --surgery_scope final_classifier \
    "${COMMON[@]}" --output_dir "$OUT" 2>&1 | tee "$LOG"

  OUT="$PROJECT/results/experiments/mechanism/ema_last_two_fc_${SEED}"
  LOG="$PROJECT/logs/experiments/mechanism/ema_last_two_fc_${SEED}.log"
  CUDA_VISIBLE_DEVICES="$GPU" python -u "$PROJECT/fedcagc_v2_mechanism_experiments.py" \
    --method fedcagc --prototype_mode ema --surgery_scope last_two_fc \
    "${COMMON[@]}" --output_dir "$OUT" 2>&1 | tee "$LOG"

  OUT="$PROJECT/results/experiments/mechanism/pcgrad_history_${SEED}"
  LOG="$PROJECT/logs/experiments/mechanism/pcgrad_history_${SEED}.log"
  CUDA_VISIBLE_DEVICES="$GPU" python -u "$PROJECT/fedcagc_v2_mechanism_experiments.py" \
    --method pcgrad_history --prototype_mode ema --surgery_scope final_classifier \
    "${COMMON[@]}" --no_wm_compensate --output_dir "$OUT" 2>&1 | tee "$LOG"
done
