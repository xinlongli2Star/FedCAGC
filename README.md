# FedCAGC: Conflict-Aware Gradient Correction for Federated Learning Watermarking

> Code and reproducibility package for **《基于冲突感知梯度修正的联邦学习水印方法》**.

FedCAGC is designed for **multi-client black-box watermarking in federated learning**. The method focuses on negative interference among client watermark gradients. It maintains historical watermark-gradient prototypes on the server and performs conflict-aware correction only in the final-classifier subspace, while keeping the global aggregation rule as standard FedAvg.

![Uploading FedCAGC_总体框架图_高仿预览.png…]()


---

## Highlights

- **Unified V2 protocol**: 10 clients, 100 communication rounds, 3 random seeds, Dirichlet Non-IID α=0.5.
- **Strict watermark train/test separation**: 100 watermark-training samples/client from the official MNIST train split and 200 watermark-test samples/client from the official MNIST test split.
- **Conflict-aware correction**: detect negative conflicts by cosine similarity, process stronger conflicts first, and re-check after each projection.
- **Low-dimensional correction space**: correction is restricted to the final classifier rather than the full model.
- **Historical prototype design**: round 1 performs normal training and initializes prototypes; correction starts from round 2.
- **Reproducibility package**: main comparison, ablation, PCGrad-history, TraMark, sensitivity, runtime/storage/communication, privacy pressure test, and fine-tuning/pruning robustness.

---

## Method overview

For client `i`, FedCAGC uses the current watermark gradient of the final classifier and the historical watermark-gradient prototypes of the other clients.

```text
Round 1
  Local main-task + watermark training
            │
            ├── upload local model ───────────────► FedAvg
            └── upload raw watermark gradient ───► initialize prototypes

Round 2+
  Global model + historical prototypes
            │
            ▼
  current watermark gradient g_i
            │
            ├── cosine similarity with other prototypes
            ├── keep negative-conflict candidates only
            ├── sort by conflict strength: most negative first
            ├── sequential projection + conflict re-check
            └── bounded norm compensation
            │
            ▼
  continue local optimization ───────────────────► FedAvg
                                                    │
                                                    └── EMA prototype update
```

The correction rule removes **negative conflict components only**. Positive overlap is not explicitly removed.

---

## Repository structure

```text
FedCAGC/
├── README.md
├── requirements.txt
├── SHA256SUMS.txt
├── fedcagc_v2_all_methods.py
├── fedcagc_v2_review_experiments.py
├── fedcagc_v2_offline_review.py
├── fedcagc_v2_robustness.py
└── fedcagc_v2_tramark.py
```

| File                               | Purpose                                                      |
| ---------------------------------- | ------------------------------------------------------------ |
| `fedcagc_v2_all_methods.py`        | Main experiments, FedCAGC, FedAvg, FedIPR, FLWB, FedAWM, ablation, sensitivity and runtime profiling |
| `fedcagc_v2_review_experiments.py` | PCGrad-history, same-round fresh-gradient reference, last-two-FC mechanism experiments |
| `fedcagc_v2_offline_review.py`     | Client-level ownership analysis, wrong-key evaluation, prototype consistency, projection-order replay, leakage pressure test, statistics |
| `fedcagc_v2_robustness.py`         | Fine-tuning and pruning robustness                           |
| `fedcagc_v2_tramark.py`            | TraMark reproduction under the unified V2 protocol           |

---

## Experimental protocol

### Federated learning setting

| Parameter            |                            Value |
| -------------------- | -------------------------------: |
| Main datasets        | Fashion-MNIST (FMNIST), CIFAR-10 |
| Clients              |                               10 |
| Communication rounds |                              100 |
| Seeds                |                 3047, 3048, 3049 |
| Data partition       |                Dirichlet Non-IID |
| Dirichlet α          |                              0.5 |
| Local epochs         |                                5 |
| Batch size           |                               64 |
| Optimizer            |                              SGD |
| Learning rate        |                             0.01 |
| Momentum             |                              0.9 |
| Weight decay         |                             1e-4 |
| Gradient clipping    |                             20.0 |
| Server aggregation   |                  weighted FedAvg |

### Watermark setting

| Parameter             |                                          Value |
| --------------------- | ---------------------------------------------: |
| Watermark source      |                                          MNIST |
| WM train              | 100 samples/client, official MNIST train split |
| WM test               |  200 samples/client, official MNIST test split |
| WM train/test overlap |                                           none |
| WM loss weight        |                                            1.0 |

### FedCAGC default setting

| Parameter                         |                                 Value |
| --------------------------------- | ------------------------------------: |
| Correction scope                  |                      final classifier |
| EMA coefficient `rho`             |                                   0.9 |
| Negative-conflict threshold `tau` |                                   0.0 |
| Candidate order                   |              initial cosine ascending |
| Re-check after projection         |                                   yes |
| Norm compensation                 |                                   yes |
| Maximum compensation scale        |                                   2.0 |
| Prototype source                  | raw / uncorrected watermark gradients |
| Prototype update                  |                EMA + L2 normalization |
| Round 1                           |             cold start, no correction |
| Round 2+                          |             conflict-aware correction |

---

## Environment

The reported experiments were conducted with:

| Item        | Configuration             |
| ----------- | ------------------------- |
| OS          | Ubuntu 22.04.4 LTS 64-bit |
| Python      | 3.8.12                    |
| PyTorch     | 1.10.1 + CUDA 11.3        |
| torchvision | 0.11.2                    |
| NumPy       | 1.21.6                    |
| GPU         | NVIDIA RTX 4090D 24 GB    |
| CPU         | AMD EPYC 9654             |
| RAM         | 755 GB                    |

Recommended installation:

```bash
conda create -n fedcagc python=3.8 -y
conda activate fedcagc
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install numpy==1.21.6 "pandas>=1.3,<2.0"
```

The datasets are downloaded automatically by torchvision on first use.

---

## Quick start

From the repository root:

```bash
export PROJECT=$(pwd)
export DATA_PATH=$PROJECT/data
mkdir -p "$DATA_PATH" results/main results/reviewer logs
```

Run one CIFAR-10 FedCAGC experiment:

```bash
SEED=3047
CUDA_VISIBLE_DEVICES=0 python -u fedcagc_v2_all_methods.py \
  --dataset cifar10 --method fedcagc \
  --data_path "$DATA_PATH" --device cuda \
  --num_clients 10 --rounds 100 --seed $SEED \
  --local_epochs 5 --local_bs 64 --local_lr 0.01 \
  --momentum 0.9 --weight_decay 1e-4 \
  --non_iid --gamma 0.5 \
  --watermark_source mnist --wm_train_size 100 --wm_test_size 200 --wm_beta 1.0 \
  --surgery_scope final_classifier --ema_rho 0.9 --tau_neg 0.0 \
  --wm_compensate --max_comp_scale 2.0 --grad_clip_norm 20 \
  --eval_every 5 --eval_tail 10 --snapshot_rounds 1,10,20,50,100 \
  --profile_round \
  --output_dir "$PROJECT/results/main/cifar10_fedcagc_$SEED"
```

For FMNIST, replace `--dataset cifar10` with `--dataset fmnist` and update the output directory. Repeat with seeds `3047`, `3048`, and `3049` for the reported three-seed results.

---

## Main comparison

The final comparison includes:

```text
FedAvg
FedIPR
FLWB
FedAWM
TraMark
FedCAGC (ours)
```

Results are reported as **mean ± SD over seeds 3047/3048/3049**.

### FMNIST

| Method      |             MTA |             WMA |          post-WGC |
| ----------- | --------------: | --------------: | ----------------: |
| FedAvg      |     91.85±0.15% |               — |                 — |
| FedIPR      |     91.69±0.10% |     89.58±1.44% |     0.0984±0.0064 |
| FLWB        |     91.82±0.23% |     78.48±2.71% |     0.0956±0.0065 |
| FedAWM      |     91.62±0.22% |     89.83±1.55% |     0.0959±0.0051 |
| TraMark     |     90.29±0.28% |     92.50±0.36% |                 — |
| **FedCAGC** | **91.69±0.05%** | **89.77±1.29%** | **0.0648±0.0016** |

### CIFAR-10

| Method      |             MTA |             WMA |          post-WGC |
| ----------- | --------------: | --------------: | ----------------: |
| FedAvg      |     87.64±0.18% |               — |                 — |
| FedIPR      |     87.08±0.45% |     90.85±3.94% |     0.0846±0.0102 |
| FLWB        |     87.24±0.27% |     76.68±3.75% |     0.1003±0.0099 |
| FedAWM      |     87.42±0.17% |     94.20±0.56% |     0.0928±0.0040 |
| TraMark     |     85.97±0.37% |     83.60±2.73% |                 — |
| **FedCAGC** | **87.24±0.43%** | **94.02±0.10%** | **0.0535±0.0095** |

> TraMark additionally achieves `VR = 100.00±0.00%` on both datasets. Its VR measures identity tracing on personalized models and is **not numerically equivalent** to WMA on the shared global model. TraMark does not use the same shared-parameter multi-client WGC definition, so WGC is reported as `—`.

---

## Reproduction map

| Paper experiment                  | Script                             | Main output directory           |
| --------------------------------- | ---------------------------------- | ------------------------------- |
| Main comparison                   | `fedcagc_v2_all_methods.py`        | `results/main/`                 |
| TraMark comparison                | `fedcagc_v2_tramark.py`            | `results/reviewer/tramark/`     |
| Ablation                          | `fedcagc_v2_all_methods.py`        | `results/reviewer/ablation/`    |
| Same-round gradient / last-two-FC | `fedcagc_v2_review_experiments.py` | `results/reviewer/mechanism/`   |
| Projection-order replay           | `fedcagc_v2_offline_review.py`     | `results/reviewer/offline/`     |
| Ownership / wrong-key             | `fedcagc_v2_offline_review.py`     | `results/reviewer/offline/`     |
| Prototype leakage                 | `fedcagc_v2_offline_review.py`     | `results/reviewer/offline/`     |
| Sensitivity                       | `fedcagc_v2_all_methods.py`        | `results/reviewer/sensitivity/` |
| PCGrad-history                    | `fedcagc_v2_review_experiments.py` | `results/reviewer/pcgrad/`      |
| Runtime profiling                 | `fedcagc_v2_all_methods.py`        | `results/reviewer/runtime/`     |
| Fine-tuning / pruning             | `fedcagc_v2_robustness.py`         | `results/reviewer/robustness/`  |

---

<details>
<summary><b>Full main-baseline commands</b></summary>


All commands below use CIFAR-10 as the example. Replace `cifar10` with `fmnist` for FMNIST and repeat each run for seeds `3047/3048/3049`.

### FedAvg

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

### FedIPR

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

### FLWB

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

### FedAWM

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

</details>

---

<details>
<summary><b>TraMark reproduction</b></summary>


TraMark uses the **same client partition and watermark train/test indices** as the corresponding FedCAGC V2 run. Therefore, run the matching FedCAGC `dataset/seed` first.

TraMark-specific parameters:

| Parameter              |                 Value |
| ---------------------- | --------------------: |
| Warmup ratio           |                   0.5 |
| Watermark-region ratio |                  0.01 |
| WM epochs              |                     5 |
| WM LR                  |                  1e-4 |
| WM momentum            |                     0 |
| WM batch size          |                    32 |
| WM gradient mode       | `official_accumulate` |
| WM transform           |            `official` |

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

Repeat for both datasets and all three seeds.

</details>

---

<details>
<summary><b>Ablation and mechanism experiments</b></summary>


### w/o norm compensation

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

### w/o EMA

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

### Same-round fresh-gradient reference

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

### Last-two-FC correction space

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

</details>

---

<details>
<summary><b>PCGrad-history mechanism baseline</b></summary>


PCGrad-history shares the historical watermark-gradient prototypes and final-classifier subspace with FedCAGC, but uses random-order PCGrad-style negative-conflict projection and disables bounded norm compensation.

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

Repeat for seeds 3047/3048/3049.

Reported CIFAR-10 result:

| Method         |         MTA |         WMA |      post-WGC |
| -------------- | ----------: | ----------: | ------------: |
| PCGrad-history | 87.30±0.36% | 94.40±0.74% | 0.6487±0.0119 |
| FedCAGC        | 87.24±0.43% | 94.02±0.10% | 0.0535±0.0095 |

PCGrad-history is a **mechanism-level adaptation** for this federated-watermark setting, not a direct reproduction of the original centralized PCGrad algorithm.

</details>

---

<details>
<summary><b>Projection-order replay, ownership verification and privacy pressure test</b></summary>


These analyses reuse artifacts from the completed FedCAGC main runs.

### Prototype consistency + projection-order replay

```bash
python fedcagc_v2_offline_review.py --project "$PROJECT" snapshot \
  --dataset cifar10 --seed 3047 --random_replays 20
```

Repeat for 3048 and 3049.

### Client-level ownership / wrong-key evaluation

```bash
python fedcagc_v2_offline_review.py --project "$PROJECT" ownership \
  --datasets cifar10 fmnist --seeds 3047 3048 3049
```

### Prototype leakage pressure test

```bash
CUDA_VISIBLE_DEVICES=0 python fedcagc_v2_offline_review.py --project "$PROJECT" leakage \
  --dataset cifar10 --seed 3047 --batch_size 64 --device cuda
```

Repeat for 3048 and 3049.

</details>

---

<details>
<summary><b>Hyperparameter sensitivity</b></summary>


All sensitivity experiments use CIFAR-10 and seeds 3047/3048/3049. One parameter is varied at a time.

| Parameter              | Values             |
| ---------------------- | ------------------ |
| EMA coefficient ρ      | 0.8, **0.9**, 0.99 |
| Conflict threshold τ   | **0**, 0.05, 0.10  |
| Max compensation scale | 1.5, **2.0**, 2.5  |

Use the standard FedCAGC command and change only the corresponding argument and output directory.

Examples:

```bash
# rho = 0.8
--ema_rho 0.8 --tau_neg 0 --max_comp_scale 2.0 \
--output_dir "$PROJECT/results/reviewer/sensitivity/rho_080_$SEED"

# tau = 0.05
--ema_rho 0.9 --tau_neg 0.05 --max_comp_scale 2.0 \
--output_dir "$PROJECT/results/reviewer/sensitivity/tau_005_$SEED"

# smax = 1.5
--ema_rho 0.9 --tau_neg 0 --max_comp_scale 1.5 \
--output_dir "$PROJECT/results/reviewer/sensitivity/smax_15_$SEED"
```

</details>

---

<details>
<summary><b>Runtime, fine-tuning and pruning robustness</b></summary>


### Runtime profiling

The reported runtime comparison uses CIFAR-10, 5 communication rounds and the same three seeds.

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

Use the same command with `--method fedipr` for the reference runtime.

### Fine-tuning + pruning

The robustness script loads the 100-round FedCAGC checkpoint from:

```text
results/main/<dataset>_fedcagc_<seed>/final_checkpoint.pt
```

```bash
SEED=3047
CUDA_VISIBLE_DEVICES=0 python -u fedcagc_v2_robustness.py \
  --project "$PROJECT" --dataset cifar10 --seed $SEED --device cuda \
  --num_workers 2 --batch_size 64 \
  --ft_epochs 50 --ft_lr 0.01 --momentum 0.9 --weight_decay 1e-4 --grad_clip_norm 20 \
  --prune_ratios 0,10,20,40,60,80,90,95,99 \
  --output_dir "$PROJECT/results/reviewer/robustness/cifar10_$SEED"
```

Repeat for FMNIST and all three seeds.

</details>

---

## Statistics

Aggregate the shared-model main experiments:

```bash
python fedcagc_v2_offline_review.py --project "$PROJECT" stats \
  --datasets cifar10 fmnist \
  --methods fedavg fedipr flwb fedawm fedcagc \
  --seeds 3047 3048 3049
```

The script outputs mean, SD and 95% confidence intervals. Same-seed WGC values are used for paired statistical comparisons where applicable.

---

## Key output files

A training run may generate:

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

Large `.pt` files are intentionally excluded by `.gitignore`; use GitHub Releases or external storage if pretrained checkpoints need to be distributed.

---

## Reproduction order

For a clean reproduction, the recommended order is:

1. FedCAGC main runs for both datasets and all three seeds.
2. FedAvg / FedIPR / FLWB / FedAWM main baselines.
3. TraMark, after the matching FedCAGC split files exist.
4. Ablation, same-round reference, last-two-FC, sensitivity, and PCGrad-history.
5. Offline ownership, prototype consistency, order replay, and leakage analyses.
6. Runtime profiling under controlled GPU occupancy.
7. Fine-tuning/pruning robustness from the final FedCAGC checkpoints.
8. Statistical aggregation.

---

## Important metric notes

- **MTA**: main-task accuracy.
- **WMA**: watermark verification accuracy.
- **Min-WMA**: minimum client-level watermark accuracy.
- **WGC**: watermark-gradient conflict measure in the shared parameter space.
- **TraMark VR** and **FedCAGC WMA** measure different objects and should not be treated as equivalent.
- TraMark does not use the same WGC definition and therefore reports WGC as `—`.
- Cross-identity wrong-key response is a sample-level response statistic; it is not automatically equivalent to thresholded claim-level FAR.
- The same-round fresh-gradient experiment is a diagnostic reference rather than a theoretical upper bound.
- The prototype leakage experiment is a pressure test under the specified attack model and does not constitute a formal privacy guarantee.
- Runtime numbers depend on hardware, framework version, and concurrent GPU load.

---

## Reproducibility checklist

Before comparing your results with the reported values, verify that:

- [ ] `dataset ∈ {fmnist, cifar10}`
- [ ] `num_clients = 10`
- [ ] `rounds = 100`
- [ ] `seed ∈ {3047, 3048, 3049}`
- [ ] `gamma = 0.5`
- [ ] `local_epochs = 5`
- [ ] `local_bs = 64`
- [ ] `local_lr = 0.01`
- [ ] `wm_train_size = 100`
- [ ] `wm_test_size = 200`
- [ ] watermark train/test come from the official MNIST train/test splits respectively
- [ ] FedCAGC uses `final_classifier`, `ema_rho=0.9`, `tau_neg=0`, `max_comp_scale=2.0`
- [ ] results are averaged over all three seeds

---

## Citation

The manuscript is currently associated with this repository. Please cite the final publication version once bibliographic information is available.

If you use one of the reproduced baselines, please also cite the corresponding original work (FedAvg, FedIPR, FLWB, FedAWM, PCGrad, and TraMark).

---

## License

No license is bundled by default. Before public reuse is encouraged, add an explicit open-source license (for example, MIT) according to the authors' and institution's requirements.

---

## Contact / Issues

For reproducibility questions, please open a GitHub Issue and include:

- dataset and seed;
- exact command;
- `config.json`;
- `run_summary.json`;
- relevant error log or traceback.

This makes reproduction issues much easier to diagnose.
