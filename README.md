# FedCAGC: Conflict-Aware Gradient Correction for Federated Learning Watermarking

本仓库提供论文 **《基于冲突感知梯度修正的联邦学习水印方法》** 的主要实验、审稿补充实验、TraMark 对比实验以及微调/剪枝鲁棒性实验代码。本文最终实验统一采用 V2 协议。

> **重要说明**
>
> 1. 最终论文主对比方法为：FedAvg、FedIPR、FLWB、FedAWM、TraMark 和 FedCAGC。
> 2. TraMark 复现实验使用与 FedCAGC 完全相同的客户端划分和水印 train/test 索引。因此，运行 TraMark 前必须先完成相同 dataset/seed 的 FedCAGC 主实验。
> 3. PCGrad-history 是面向本研究联邦水印场景构造的 **机制级适配基线**，不是原始 PCGrad 的直接复现。

---

## 1. Repository structure

```text
FedCAGC_GitHub_release/
├── README.md
├── requirements.txt
├── fedcagc_v2_all_methods.py          # 主实验 + FedCAGC + 主要基线 + 消融/敏感性
├── fedcagc_v2_review_experiments.py   # 审稿机制实验 + PCGrad-history
├── fedcagc_v2_offline_review.py       # 所有权、顺序重放、原型一致性、泄露、统计
├── fedcagc_v2_robustness.py           # 微调/剪枝鲁棒性
└── fedcagc_v2_tramark.py              # TraMark (ICLR 2026) 统一协议复现
```

运行后建议使用以下目录结构：

```text
results/
├── main/
└── reviewer/
    ├── ablation/
    ├── mechanism/
    ├── offline/
    ├── pcgrad/
    ├── robustness/
    ├── runtime/
    ├── sensitivity/
    └── tramark/
```

---

## 2. Environment

### 2.1 Original experimental environment

| Item | Configuration |
|---|---|
| OS | Ubuntu 22.04.4 LTS 64-bit |
| Python | 3.8.12 |
| PyTorch | 1.10.1 + CUDA 11.3 |
| NumPy | 1.21.6 |
| GPU | NVIDIA RTX 4090D 24 GB |
| CPU | AMD EPYC 9654 |
| RAM | 755 GB |

PyTorch 1.10.1 官方对应 torchvision 0.11.2。GPU 环境可使用：

```bash
conda create -n fedcagc python=3.8 -y
conda activate fedcagc
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install numpy==1.21.6 "pandas>=1.3,<2.0"
```

也可使用：

```bash
pip install -r requirements.txt
```

> 若使用较新 PyTorch/CUDA 版本，数值结果可能存在轻微差异。论文结果以以上原始实验环境为准。

---

## 3. Unified V2 experimental protocol

### 3.1 Federated learning setting

| Parameter | Value |
|---|---:|
| Main datasets | Fashion-MNIST (FMNIST), CIFAR-10 |
| Clients | 10 |
| Communication rounds | 100 |
| Seeds | 3047, 3048, 3049 |
| Partition | Dirichlet Non-IID |
| Dirichlet parameter | α = 0.5 (`--gamma 0.5`) |
| Local epochs | 5 |
| Batch size | 64 |
| Optimizer | SGD |
| Learning rate | 0.01 |
| Momentum | 0.9 |
| Weight decay | 1e-4 |
| Gradient clipping | 20.0 |
| Aggregation | weighted FedAvg |
| Eval rounds | round 1, every 5 rounds, and every round in 91–100 |
| Snapshot rounds | 1, 10, 20, 50, 100 |

### 3.2 Watermark protocol

| Parameter | Value |
|---|---:|
| Watermark source | MNIST |
| WM train | 100 samples/client from official MNIST train split |
| WM test | 200 samples/client from official MNIST test split |
| Train/test overlap | none |
| FMNIST watermark shape | 1 × 28 × 28 |
| CIFAR-10 watermark shape | 3 × 32 × 32 |
| WM loss weight | 1.0 |

### 3.3 FedCAGC default configuration

| Parameter | Value |
|---|---:|
| Surgery scope | final classifier |
| EMA coefficient ρ | 0.9 |
| Negative-conflict threshold τ | 0.0 |
| Conflict candidate order | initial cosine ascending (most negative first) |
| Re-check | yes, after every projection |
| Positive-overlap removal | no |
| Norm compensation | yes |
| Max compensation scale | 2.0 |
| Prototype source | raw / uncorrected watermark gradients |
| Prototype update | EMA then L2 normalization |
| Round 1 | cold start, no correction |
| Round 2+ | conflict-aware correction enabled |

---

## 4. Basic preparation

From the repository root:

```bash
export PROJECT=$(pwd)
export DATA_PATH=$PROJECT/data
mkdir -p "$DATA_PATH" results/main results/reviewer logs
```

Datasets are downloaded automatically by torchvision when first used.

For GPU selection, prepend commands with e.g.:

```bash
CUDA_VISIBLE_DEVICES=0
```

All commands below use `--device cuda`. For CPU-only debugging, use `--device cpu`.

---

## 5. Main experiments

All main runs use the following common V2 configuration:

```text
--num_clients 10
--rounds 100
--local_epochs 5
--local_bs 64
--local_lr 0.01
--momentum 0.9
--weight_decay 1e-4
--non_iid
--gamma 0.5
--watermark_source mnist
--wm_train_size 100
--wm_test_size 200
--wm_beta 1.0
--grad_clip_norm 20
--eval_every 5
--eval_tail 10
--snapshot_rounds 1,10,20,50,100
--profile_round
```

For every method/dataset pair, repeat the command with `SEED=3047`, `SEED=3048`, and `SEED=3049`.

### 5.1 FedAvg

```bash
SEED=3047
CUDA_VISIBLE_DEVICES=0 python -u fedcagc_v2_all_methods.py \
  --dataset cifar10 --method fedavg --data_path "$DATA_PATH" --device cuda \
  --num_clients 10 --rounds 100 --seed $SEED --local_epochs 5 --local_bs 64 --local_lr 0.01 \
  --momentum 0.9 --weight_decay 1e-4 --non_iid --gamma 0.5 \
  --watermark_source mnist --wm_train_size 100 --wm_test_size 200 --wm_beta 1.0 \
  --grad_clip_norm 20 --eval_every 5 --eval_tail 10 --snapshot_rounds 1,10,20,50,100 \
  --profile_round --output_dir "$PROJECT/results/main/cifar10_fedavg_$SEED"
```

FMNIST only changes `--dataset fmnist` and the output directory to `fmnist_fedavg_$SEED`.

### 5.2 FedIPR

```bash
SEED=3047
CUDA_VISIBLE_DEVICES=0 python -u fedcagc_v2_all_methods.py \
  --dataset cifar10 --method fedipr --data_path "$DATA_PATH" --device cuda \
  --num_clients 10 --rounds 100 --seed $SEED --local_epochs 5 --local_bs 64 --local_lr 0.01 \
  --momentum 0.9 --weight_decay 1e-4 --non_iid --gamma 0.5 \
  --watermark_source mnist --wm_train_size 100 --wm_test_size 200 --wm_beta 1.0 \
  --grad_clip_norm 20 --eval_every 5 --eval_tail 10 --snapshot_rounds 1,10,20,50,100 \
  --profile_round --output_dir "$PROJECT/results/main/cifar10_fedipr_$SEED"
```

### 5.3 FLWB

Formal setting: `flwb_lambda=1.0`, `flwb_wm_steps=1`.

```bash
SEED=3047
CUDA_VISIBLE_DEVICES=0 python -u fedcagc_v2_all_methods.py \
  --dataset cifar10 --method flwb --data_path "$DATA_PATH" --device cuda \
  --num_clients 10 --rounds 100 --seed $SEED --local_epochs 5 --local_bs 64 --local_lr 0.01 \
  --momentum 0.9 --weight_decay 1e-4 --non_iid --gamma 0.5 \
  --watermark_source mnist --wm_train_size 100 --wm_test_size 200 --wm_beta 1.0 \
  --flwb_lambda 1.0 --flwb_wm_steps 1 --grad_clip_norm 20 \
  --eval_every 5 --eval_tail 10 --snapshot_rounds 1,10,20,50,100 \
  --profile_round --output_dir "$PROJECT/results/main/cifar10_flwb_$SEED"
```

### 5.4 FedAWM

Formal project reproduction parameters: temperature=1.0, min_scale=0.5, max_scale=2.0, EMA=0.8.

```bash
SEED=3047
CUDA_VISIBLE_DEVICES=0 python -u fedcagc_v2_all_methods.py \
  --dataset cifar10 --method fedawm --data_path "$DATA_PATH" --device cuda \
  --num_clients 10 --rounds 100 --seed $SEED --local_epochs 5 --local_bs 64 --local_lr 0.01 \
  --momentum 0.9 --weight_decay 1e-4 --non_iid --gamma 0.5 \
  --watermark_source mnist --wm_train_size 100 --wm_test_size 200 --wm_beta 1.0 \
  --awm_temperature 1.0 --awm_min_scale 0.5 --awm_max_scale 2.0 --awm_ema 0.8 \
  --grad_clip_norm 20 --eval_every 5 --eval_tail 10 --snapshot_rounds 1,10,20,50,100 \
  --profile_round --output_dir "$PROJECT/results/main/cifar10_fedawm_$SEED"
```

### 5.5 FedCAGC

```bash
SEED=3047
CUDA_VISIBLE_DEVICES=0 python -u fedcagc_v2_all_methods.py \
  --dataset cifar10 --method fedcagc --data_path "$DATA_PATH" --device cuda \
  --num_clients 10 --rounds 100 --seed $SEED --local_epochs 5 --local_bs 64 --local_lr 0.01 \
  --momentum 0.9 --weight_decay 1e-4 --non_iid --gamma 0.5 \
  --watermark_source mnist --wm_train_size 100 --wm_test_size 200 --wm_beta 1.0 \
  --surgery_scope final_classifier --ema_rho 0.9 --tau_neg 0.0 \
  --wm_compensate --max_comp_scale 2.0 --grad_clip_norm 20 \
  --eval_every 5 --eval_tail 10 --snapshot_rounds 1,10,20,50,100 \
  --profile_round --output_dir "$PROJECT/results/main/cifar10_fedcagc_$SEED"
```

For FMNIST, replace `cifar10` with `fmnist` in `--dataset` and output directory. Repeat all main experiments for seeds 3047/3048/3049.

---

## 6. TraMark (ICLR 2026) reproduction

TraMark uses the **exact FedCAGC V2 client partition and watermark indices** for the same dataset/seed. Therefore run the corresponding FedCAGC main experiment first.

Formal TraMark-specific configuration:

| Parameter | Value |
|---|---:|
| Warmup ratio α | 0.5 |
| Watermark region ratio k | 0.01 |
| WM epochs | 5 |
| WM LR | 1e-4 |
| WM momentum | 0 |
| WM batch size | 32 |
| WM gradient mode | `official_accumulate` |
| WM transform | `official` |

CIFAR-10 example:

```bash
SEED=3047
CUDA_VISIBLE_DEVICES=0 python -u fedcagc_v2_tramark.py \
  --project "$PROJECT" --data_path "$DATA_PATH" --dataset cifar10 --seed $SEED --device cuda \
  --num_clients 10 --rounds 100 --local_epochs 5 --local_bs 64 --local_lr 0.01 \
  --momentum 0.9 --weight_decay 1e-4 --gamma 0.5 \
  --wm_train_size 100 --wm_test_size 200 \
  --alpha 0.5 --k 0.01 --wm_epochs 5 --wm_lr 0.0001 --wm_momentum 0 --wm_bs 32 \
  --wm_grad_mode official_accumulate --wm_transform official \
  --eval_every 5 --eval_tail 10 \
  --output_dir "$PROJECT/results/reviewer/tramark/cifar10_$SEED"
```

Repeat for FMNIST and seeds 3047/3048/3049.

TraMark outputs MTA, WMA, Cross-WMA and Verification Rate (VR). WGC is not reported because TraMark maintains personalized watermark parameter regions instead of one shared multi-watermark global model.

---

## 7. Ablation experiments

Ablation experiments are run on CIFAR-10 with seeds 3047/3048/3049.

### 7.1 w/o norm compensation

```bash
SEED=3047
CUDA_VISIBLE_DEVICES=0 python -u fedcagc_v2_all_methods.py \
  --dataset cifar10 --method fedcagc --data_path "$DATA_PATH" --device cuda \
  --num_clients 10 --rounds 100 --seed $SEED --local_epochs 5 --local_bs 64 --local_lr 0.01 \
  --momentum 0.9 --weight_decay 1e-4 --non_iid --gamma 0.5 \
  --watermark_source mnist --wm_train_size 100 --wm_test_size 200 --wm_beta 1.0 \
  --surgery_scope final_classifier --ema_rho 0.9 --tau_neg 0.0 \
  --no_wm_compensate --max_comp_scale 2.0 --grad_clip_norm 20 \
  --eval_every 5 --eval_tail 10 --snapshot_rounds 1,10,20,50,100 --profile_round \
  --output_dir "$PROJECT/results/reviewer/ablation/wocomp_$SEED"
```

### 7.2 w/o EMA smoothing

```bash
SEED=3047
CUDA_VISIBLE_DEVICES=0 python -u fedcagc_v2_all_methods.py \
  --dataset cifar10 --method fedcagc --data_path "$DATA_PATH" --device cuda \
  --num_clients 10 --rounds 100 --seed $SEED --local_epochs 5 --local_bs 64 --local_lr 0.01 \
  --momentum 0.9 --weight_decay 1e-4 --non_iid --gamma 0.5 \
  --watermark_source mnist --wm_train_size 100 --wm_test_size 200 --wm_beta 1.0 \
  --surgery_scope final_classifier --ema_rho 0.0 --tau_neg 0.0 \
  --wm_compensate --max_comp_scale 2.0 --grad_clip_norm 20 \
  --eval_every 5 --eval_tail 10 --snapshot_rounds 1,10,20,50,100 --profile_round \
  --output_dir "$PROJECT/results/reviewer/ablation/woema_$SEED"
```

The Full FedCAGC row is the corresponding main experiment and should not be independently redefined with another protocol.

---

## 8. EMA prototype and correction-space mechanism experiments

Run all three seeds on CIFAR-10.

### 8.1 Same-round fresh-gradient reference

```bash
SEED=3047
CUDA_VISIBLE_DEVICES=0 python -u fedcagc_v2_review_experiments.py \
  --dataset cifar10 --method fedcagc --data_path "$DATA_PATH" --device cuda \
  --num_clients 10 --rounds 100 --seed $SEED --local_epochs 5 --local_bs 64 --local_lr 0.01 \
  --momentum 0.9 --weight_decay 1e-4 --non_iid --gamma 0.5 \
  --watermark_source mnist --wm_train_size 100 --wm_test_size 200 --wm_beta 1.0 \
  --surgery_scope final_classifier --prototype_mode fresh_round_start \
  --ema_rho 0.9 --tau_neg 0.0 --wm_compensate --max_comp_scale 2.0 --grad_clip_norm 20 \
  --eval_every 5 --eval_tail 10 --snapshot_rounds 1,10,20,50,100 --profile_round \
  --output_dir "$PROJECT/results/reviewer/mechanism/fresh_round_start_$SEED"
```

This is an oracle-like diagnostic reference, not a theoretical upper bound.

### 8.2 Last-two-FC correction space

```bash
SEED=3047
CUDA_VISIBLE_DEVICES=0 python -u fedcagc_v2_review_experiments.py \
  --dataset cifar10 --method fedcagc --data_path "$DATA_PATH" --device cuda \
  --num_clients 10 --rounds 100 --seed $SEED --local_epochs 5 --local_bs 64 --local_lr 0.01 \
  --momentum 0.9 --weight_decay 1e-4 --non_iid --gamma 0.5 \
  --watermark_source mnist --wm_train_size 100 --wm_test_size 200 --wm_beta 1.0 \
  --surgery_scope last_two_fc --prototype_mode ema \
  --ema_rho 0.9 --tau_neg 0.0 --wm_compensate --max_comp_scale 2.0 --grad_clip_norm 20 \
  --eval_every 5 --eval_tail 10 --snapshot_rounds 1,10,20,50,100 --profile_round \
  --output_dir "$PROJECT/results/reviewer/mechanism/last_two_fc_$SEED"
```

---

## 9. Projection-order replay and prototype consistency

These analyses use the saved `prototype_snapshots.pt` and `gradient_snapshots.pt` from the FedCAGC main runs. They do not retrain the federated model.

```bash
python fedcagc_v2_offline_review.py --project "$PROJECT" snapshot --dataset cifar10 --seed 3047 --random_replays 20
python fedcagc_v2_offline_review.py --project "$PROJECT" snapshot --dataset cifar10 --seed 3048 --random_replays 20
python fedcagc_v2_offline_review.py --project "$PROJECT" snapshot --dataset cifar10 --seed 3049 --random_replays 20
```

Outputs:

```text
results/reviewer/offline/cifar10_seed<seed>_prototype_consistency.csv
results/reviewer/offline/cifar10_seed<seed>_order_replay.csv
```

The order replay compares conflict-strength order, fixed Client-ID order, and 20 random orders on fixed saved gradient/prototype snapshots.

---

## 10. Client-level ownership / wrong-key evaluation

Requires FedCAGC and FedAvg main results for both datasets and all three seeds.

```bash
python fedcagc_v2_offline_review.py --project "$PROJECT" ownership \
  --datasets cifar10 fmnist --seeds 3047 3048 3049
```

This produces client-level correct-key / cross-identity wrong-key statistics. The reported cross-identity response is a sample-level response statistic, not a thresholded claim-level FAR.

---

## 11. Prototype leakage pressure test

The leakage analysis uses the final FedCAGC checkpoint and historical prototypes. Formal experiments use CIFAR-10 and seeds 3047/3048/3049.

```bash
CUDA_VISIBLE_DEVICES=0 python fedcagc_v2_offline_review.py --project "$PROJECT" leakage --dataset cifar10 --seed 3047 --batch_size 64 --device cuda
CUDA_VISIBLE_DEVICES=0 python fedcagc_v2_offline_review.py --project "$PROJECT" leakage --dataset cifar10 --seed 3048 --batch_size 64 --device cuda
CUDA_VISIBLE_DEVICES=0 python fedcagc_v2_offline_review.py --project "$PROJECT" leakage --dataset cifar10 --seed 3049 --batch_size 64 --device cuda
```

Outputs include overall ROC-AUC, best balanced accuracy, per-client statistics and raw attack scores.

---

## 12. Hyperparameter sensitivity

All sensitivity experiments are CIFAR-10, seeds 3047/3048/3049, and vary one parameter at a time. The center/default configuration (`rho=0.9`, `tau=0`, `smax=2.0`) reuses the main FedCAGC run.

### 12.1 Grid

| Parameter | Values |
|---|---|
| EMA ρ | 0.8, **0.9**, 0.99 |
| τ | **0**, 0.05, 0.10 |
| smax | 1.5, **2.0**, 2.5 |

### 12.2 rho = 0.8

```bash
SEED=3047
CUDA_VISIBLE_DEVICES=0 python -u fedcagc_v2_all_methods.py \
  --dataset cifar10 --method fedcagc --data_path "$DATA_PATH" --device cuda \
  --num_clients 10 --rounds 100 --seed $SEED --local_epochs 5 --local_bs 64 --local_lr 0.01 \
  --momentum 0.9 --weight_decay 1e-4 --non_iid --gamma 0.5 \
  --watermark_source mnist --wm_train_size 100 --wm_test_size 200 --wm_beta 1.0 \
  --ema_rho 0.8 --tau_neg 0 --wm_compensate --max_comp_scale 2.0 --grad_clip_norm 20 \
  --eval_every 5 --eval_tail 10 --snapshot_rounds 1,10,20,50,100 --profile_round \
  --output_dir "$PROJECT/results/reviewer/sensitivity/rho_080_$SEED"
```

For the remaining settings use the same command and change only:

| Experiment | Changed argument | Output directory |
|---|---|---|
| rho=0.99 | `--ema_rho 0.99` | `rho_099_<seed>` |
| tau=0.05 | `--tau_neg 0.05` | `tau_005_<seed>` |
| tau=0.10 | `--tau_neg 0.10` | `tau_010_<seed>` |
| smax=1.5 | `--max_comp_scale 1.5` | `smax_15_<seed>` |
| smax=2.5 | `--max_comp_scale 2.5` | `smax_25_<seed>` |

Repeat each for 3047, 3048 and 3049.

---

## 13. PCGrad-history mechanism baseline

PCGrad-history uses the same historical watermark-gradient prototypes and final-classifier subspace as FedCAGC, but applies PCGrad-style random-order negative-conflict projection and disables norm compensation.

```bash
SEED=3047
CUDA_VISIBLE_DEVICES=0 python -u fedcagc_v2_review_experiments.py \
  --dataset cifar10 --method pcgrad_history --data_path "$DATA_PATH" --device cuda \
  --num_clients 10 --rounds 100 --seed $SEED --local_epochs 5 --local_bs 64 --local_lr 0.01 \
  --momentum 0.9 --weight_decay 1e-4 --non_iid --gamma 0.5 \
  --watermark_source mnist --wm_train_size 100 --wm_test_size 200 --wm_beta 1.0 \
  --surgery_scope final_classifier --prototype_mode ema --ema_rho 0.9 --tau_neg 0.0 \
  --no_wm_compensate --max_comp_scale 2.0 --grad_clip_norm 20 \
  --eval_every 5 --eval_tail 10 --snapshot_rounds 1,10,20,50,100 --profile_round \
  --output_dir "$PROJECT/results/reviewer/pcgrad/pcgrad_history_$SEED"
```

Repeat for seeds 3047, 3048 and 3049.

The paper reports PCGrad-history as a mechanism-level adapted baseline, not as the original centralized PCGrad algorithm.

---

## 14. Runtime profiling

The reported runtime experiment uses CIFAR-10, 5 communication rounds and three seeds. Evaluation is performed every round. Run FedIPR and FedCAGC separately under the same hardware/process conditions.

### 14.1 FedIPR

```bash
SEED=3047
CUDA_VISIBLE_DEVICES=0 python -u fedcagc_v2_all_methods.py \
  --dataset cifar10 --method fedipr --data_path "$DATA_PATH" --device cuda \
  --num_clients 10 --rounds 5 --seed $SEED --local_epochs 5 --local_bs 64 --local_lr 0.01 \
  --momentum 0.9 --weight_decay 1e-4 --non_iid --gamma 0.5 \
  --watermark_source mnist --wm_train_size 100 --wm_test_size 200 --wm_beta 1.0 \
  --grad_clip_norm 20 --eval_every 1 --eval_tail 0 --snapshot_rounds 1,5 --profile_round \
  --output_dir "$PROJECT/results/reviewer/runtime/fedipr_5round_$SEED"
```

### 14.2 FedCAGC

```bash
SEED=3047
CUDA_VISIBLE_DEVICES=0 python -u fedcagc_v2_all_methods.py \
  --dataset cifar10 --method fedcagc --data_path "$DATA_PATH" --device cuda \
  --num_clients 10 --rounds 5 --seed $SEED --local_epochs 5 --local_bs 64 --local_lr 0.01 \
  --momentum 0.9 --weight_decay 1e-4 --non_iid --gamma 0.5 \
  --watermark_source mnist --wm_train_size 100 --wm_test_size 200 --wm_beta 1.0 \
  --ema_rho 0.9 --tau_neg 0 --wm_compensate --max_comp_scale 2.0 --grad_clip_norm 20 \
  --eval_every 1 --eval_tail 0 --snapshot_rounds 1,5 --profile_round \
  --output_dir "$PROJECT/results/reviewer/runtime/fedcagc_5round_$SEED"
```

Repeat both methods for seeds 3047/3048/3049. Runtime numbers are implementation- and hardware-dependent and should not be treated as hardware-independent constants.

---

## 15. Fine-tuning and pruning robustness

This script does not retrain federated learning. It loads the corresponding 100-round FedCAGC checkpoint from:

```text
results/main/<dataset>_fedcagc_<seed>/final_checkpoint.pt
```

Formal settings:

| Fine-tuning | Value |
|---|---:|
| Data | clean main-task train data only |
| WM samples | not used |
| Epochs | 50 |
| Batch | 64 |
| LR | 0.01 |
| Momentum | 0.9 |
| Weight decay | 1e-4 |
| Gradient clipping | 20 |

| Pruning | Value |
|---|---|
| Method | global unstructured magnitude pruning |
| Scope | Conv2d / Linear weights |
| Ratios (%) | 0, 10, 20, 40, 60, 80, 90, 95, 99 |
| Post-pruning fine-tune | no |
| Checkpoint handling | reload original checkpoint for every pruning ratio |

CIFAR-10 example:

```bash
SEED=3047
CUDA_VISIBLE_DEVICES=0 python -u fedcagc_v2_robustness.py \
  --project "$PROJECT" --dataset cifar10 --seed $SEED --device cuda \
  --num_workers 2 --batch_size 64 --ft_epochs 50 --ft_lr 0.01 \
  --momentum 0.9 --weight_decay 1e-4 --grad_clip_norm 20 \
  --prune_ratios 0,10,20,40,60,80,90,95,99 \
  --output_dir "$PROJECT/results/reviewer/robustness/cifar10_$SEED"
```

Repeat for FMNIST and seeds 3047/3048/3049.

---

## 16. Mean ± SD and 95% confidence intervals

For the five shared-model main methods:

```bash
python fedcagc_v2_offline_review.py --project "$PROJECT" stats \
  --datasets cifar10 fmnist \
  --methods fedavg fedipr flwb fedawm fedcagc \
  --seeds 3047 3048 3049
```

This writes:

```text
results/reviewer/offline/main_seed_metrics.csv
results/reviewer/offline/main_mean_sd_ci95.csv
```

TraMark results are stored separately under `results/reviewer/tramark/<dataset>_<seed>/run_summary.json` because TraMark uses personalized models and has an additional VR metric.

The paired t-test values reported in the manuscript were computed on same-seed WGC measurements. With n=3, statistical significance is treated as auxiliary evidence rather than a standalone conclusion.

---

## 17. Key output files

Each main/reviewer training run typically produces:

```text
config.json
client_partition.json
watermark_split.json
round_metrics.csv
verification_matrix.csv
run_summary.json
final_checkpoint.pt
prototype_snapshots.pt
gradient_snapshots.pt
```

Depending on the method, some artifacts may be absent or not applicable.

For public GitHub release, large `.pt` checkpoints should normally be distributed via a separate release/archive rather than committed directly to Git history.

---

## 18. Reproduction dependency order

Recommended order:

1. Run all FedCAGC main experiments first.
2. Run FedAvg/FedIPR/FLWB/FedAWM main baselines.
3. Run TraMark after the matching FedCAGC split files exist.
4. Run ablation, fresh-gradient, last-two-FC, sensitivity and PCGrad-history experiments.
5. Run offline ownership / prototype consistency / order replay / leakage analyses.
6. Run 5-round runtime profiling under controlled hardware occupancy.
7. Run fine-tuning/pruning robustness from the 100-round FedCAGC checkpoints.
8. Run statistics aggregation.

---

## 19. Paper-result validation targets

These values are provided only as sanity checks for the final V2 setup; small hardware/software numerical differences are possible.

### FedCAGC, round 100

| Dataset | MTA | WMA | pre-WGC | post-WGC |
|---|---:|---:|---:|---:|
| FMNIST | 91.69 ± 0.05% | 89.77 ± 1.29% | 0.0975 ± 0.0040 | 0.0648 ± 0.0016 |
| CIFAR-10 | 87.24 ± 0.43% | 94.02 ± 0.10% | 0.0922 ± 0.0025 | 0.0535 ± 0.0095 |

### TraMark, round 100

| Dataset | MTA | WMA | VR |
|---|---:|---:|---:|
| FMNIST | 90.29 ± 0.28% | 92.50 ± 0.36% | 100.00 ± 0.00% |
| CIFAR-10 | 85.97 ± 0.37% | 83.60 ± 2.73% | 100.00 ± 0.00% |

### PCGrad-history, CIFAR-10

| Method | MTA | WMA | post-WGC |
|---|---:|---:|---:|
| PCGrad-history | 87.30 ± 0.36% | 94.40 ± 0.74% | 0.6487 ± 0.0119 |
| FedCAGC | 87.24 ± 0.43% | 94.02 ± 0.10% | 0.0535 ± 0.0095 |

---

## 20. Notes on interpretation

- TraMark VR and FedCAGC WMA are different metrics and should not be treated as numerically equivalent.
- TraMark WGC is N/A because its personalized watermark regions do not match the shared-global-model WGC definition used by FedCAGC.
- The same-round fresh-gradient experiment is a diagnostic reference, not a mathematically proven upper bound.
- Prototype leakage experiments are pressure tests under the specified attack model and do not constitute a formal privacy guarantee.
- Cross-identity wrong-key response is a sample-level response statistic; it is not automatically equivalent to thresholded claim-level FAR.
- Runtime results reflect the specific PyTorch implementation and hardware environment.
- The final paper does not use the old 70-round protocol, old watermark train/test split, or old scalability/heterogeneity results.

---

## 21. References used by the experimental code

- FedAvg: McMahan et al., AISTATS 2017.
- FedIPR: ownership verification for federated deep neural network models, IEEE TPAMI 2022.
- FLWB: federated learning watermark based on model backdoor, Journal of Software 2024.
- FedAWM: adaptive watermark allocation in Non-IID federated learning, Knowledge-Based Systems 2025.
- PCGrad: Yu et al., *Gradient Surgery for Multi-Task Learning*, NeurIPS 2020.
- TraMark: Xu et al., *Traceable Black-Box Watermarks for Federated Learning*, ICLR 2026.

When using this repository in a publication, please cite the corresponding original baseline papers in addition to the FedCAGC paper.
