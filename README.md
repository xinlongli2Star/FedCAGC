# FedCAGC: Conflict-Aware Gradient Correction for Federated Learning Watermarking

Official reproduction code for **“基于冲突感知梯度修正的联邦学习水印方法”**.

FedCAGC addresses negative interference among multiple client watermarks in synchronous federated learning. Conflict detection and correction are performed only in the final-classifier watermark-gradient subspace. The server maintains one historical watermark-gradient prototype per client using EMA, while model aggregation remains weighted FedAvg.

## Repository

```text
FedCAGC/
├── README.md
├── README_zh.md
├── requirements.txt
├── fedcagc_v2_all_methods.py
├── fedcagc_v2_mechanism_experiments.py
├── fedcagc_v2_analysis.py
├── fedcagc_v2_robustness.py
├── fedcagc_v2_tramark.py
├── fedcagc_v2_scalability.py
└── scripts/
    ├── run_main_3seeds.sh
    ├── run_tramark_3seeds.sh
    ├── run_scalability_3seeds.sh
    ├── run_mechanism_3seeds.sh
    └── run_robustness_3seeds.sh
```

The repository contains source code and reproduction instructions. Runtime outputs are written to `results/` and `logs/`.

## FedCAGC mechanism

```text
Round 1
  local task + watermark training
       ├── local model ------------------------------> FedAvg
       └── raw final-classifier watermark gradient -> initialize prototypes

Round 2+
  global model + historical watermark-gradient prototypes
       │
       ▼
  current watermark gradient g_i
       ├── compute cosine similarity with other clients' historical prototypes
       ├── select negative-conflict directions
       ├── process directions in ascending cosine-similarity order
       ├── re-check the remaining directions after each projection
       └── bounded norm compensation
       │
       ▼
  continue local optimization ----------------------> FedAvg
                                                     └── EMA prototype update
```

## Experimental protocol

| Parameter | Setting |
|---|---|
| Main datasets | Fashion-MNIST (FMNIST), CIFAR-10 |
| Clients | 10 |
| Communication rounds | 100 |
| Seeds | 3047, 3048, 3049 |
| Data partition | Dirichlet non-IID, α=0.5 |
| Local epochs | 5 |
| Batch size | 64 |
| Optimizer | SGD |
| Learning rate | 0.01 |
| Momentum | 0.9 |
| Weight decay | 1e-4 |
| Gradient clipping | 20.0 |
| Server aggregation | weighted FedAvg |
| Watermark source | official MNIST |
| WM train | 100 samples/client from the official MNIST train split |
| WM test | 200 samples/client from the official MNIST test split |
| Train/test overlap | none |

Default FedCAGC parameters: `rho=0.9`, `tau=0`, `smax=2.0`.

### Reporting convention

- MTA, WMA, pre-WGC and post-WGC are reported as mean±sample-SD over seeds 3047/3048/3049.
- WGC is displayed to four decimal places.
- WGC reduction is computed from the displayed mean values:
  `(mean(pre-WGC) - mean(post-WGC)) / mean(pre-WGC) × 100%`.
- FedAvg is an unwatermarked reference; watermark-related metrics are reported as N/A.
- TraMark is compared on MTA and WMA. Its personalized watermark regions do not correspond to the shared-parameter WGC definition used by FedCAGC.

## Paper reference results

### Main comparison

| Dataset | Method | MTA | WMA | pre-WGC | post-WGC |
|---|---|---:|---:|---:|---:|
| FMNIST | FedAvg | 91.85±0.15% | — | — | — |
| FMNIST | FedIPR | 91.69±0.10% | 89.58±1.44% | 0.0984±0.0064 | 0.0984±0.0064 |
| FMNIST | FLWB | 91.82±0.23% | 78.48±2.71% | 0.0956±0.0065 | 0.0956±0.0065 |
| FMNIST | FedAWM | 91.62±0.22% | 89.83±1.55% | 0.0959±0.0051 | 0.0959±0.0051 |
| FMNIST | TraMark | 90.29±0.28% | 92.50±0.36% | — | — |
| FMNIST | FedCAGC | 91.69±0.05% | 92.60±1.30% | 0.0975±0.0040 | 0.0648±0.0016 |
| CIFAR-10 | FedAvg | 87.40±0.43% | — | — | — |
| CIFAR-10 | FedIPR | 87.08±0.45% | 90.85±3.94% | 0.0846±0.0102 | 0.0846±0.0102 |
| CIFAR-10 | FLWB | 87.24±0.27% | 76.68±3.75% | 0.1003±0.0099 | 0.1003±0.0099 |
| CIFAR-10 | FedAWM | 87.42±0.17% | 94.20±0.56% | 0.0928±0.0040 | 0.0928±0.0040 |
| CIFAR-10 | TraMark | 85.97±0.37% | 83.60±2.73% | — | — |
| CIFAR-10 | FedCAGC | 87.24±0.43% | 96.67±0.15% | 0.0922±0.0025 | 0.0535±0.0095 |

### 30/50-client scalability

| Clients | Method | MTA | WMA | pre-WGC | post-WGC | WGC reduction |
|---:|---|---:|---:|---:|---:|---:|
| 30 | FedAvg | 85.12±0.63% | — | — | — | — |
| 30 | FedCAGC | 84.42±0.48% | 98.89±1.92% | 0.0317±0.0020 | 0.0194±0.0022 | 38.80% |
| 50 | FedAvg | 82.39±0.49% | — | — | — | — |
| 50 | FedCAGC | 79.78±0.38% | 69.38±6.91% | 0.0167±0.0007 | 0.0140±0.0018 | 16.17% |

### Fine-tuning and pruning robustness

| Dataset | Initial WMA | 50-epoch fine-tuning | 60% pruning | 80% pruning | 90% pruning | 95% pruning | 99% pruning |
|---|---:|---:|---:|---:|---:|---:|---:|
| FMNIST | 92.60±1.30% | 85.45±5.71% | 91.85±1.26% | 87.10±1.78% | 66.97±8.16% | 36.70±4.80% | 14.55±3.48% |
| CIFAR-10 | 96.67±0.15% | 88.93±0.76% | 96.62±0.10% | 95.83±0.56% | 94.12±1.50% | 62.92±2.11% | 12.70±0.00% |


### Mechanism comparison

| Setting | MTA | WMA | pre-WGC | post-WGC | WGC reduction |
|---|---:|---:|---:|---:|---:|
| EMA + final classifier | 87.24±0.43% | 96.67±0.15% | 0.0922±0.0025 | 0.0535±0.0095 | 41.97% |
| Same-round current gradient + final classifier | 87.16±0.33% | 96.77±0.24% | 0.0891±0.0083 | 0.0444±0.0086 | 50.17% |
| EMA + last two FC layers | 87.47±0.31% | 96.78±0.48% | 0.0864±0.0087 | 0.0486±0.0018 | 43.75% |

### Projection-order comparison

| Order | pre-WGC | post-WGC | WGC reduction |
|---|---:|---:|---:|
| Cosine similarity ascending | 0.0883±0.0026 | 0.0535±0.0044 | 39.41% |
| Fixed client-ID order | 0.0883±0.0026 | 0.0592±0.0057 | 32.96% |
| Random order | 0.0883±0.0026 | 0.0626±0.0056 | 29.11% |

### PCGrad-history comparison

| Method | MTA | WMA | pre-WGC | post-WGC | WGC reduction |
|---|---:|---:|---:|---:|---:|
| PCGrad-history | 87.30±0.36% | 94.40±0.74% | 0.0892±0.0023 | 0.0613±0.0119 | 31.28% |
| FedCAGC | 87.24±0.43% | 96.67±0.15% | 0.0922±0.0025 | 0.0535±0.0095 | 41.97% |

### Ablation

| Method | MTA | WMA | pre-WGC | post-WGC | WGC reduction |
|---|---:|---:|---:|---:|---:|
| FedIPR | 87.08±0.45% | 90.85±3.94% | 0.0846±0.0102 | 0.0846±0.0102 | 0.00% |
| FedCAGC w/o Comp | 87.36±0.45% | 96.60±0.53% | 0.0934±0.0059 | 0.0562±0.0079 | 39.83% |
| FedCAGC w/o EMA | 87.04±0.19% | 95.68±0.78% | 0.0896±0.0027 | 0.0670±0.0168 | 25.22% |
| FedCAGC | 87.24±0.43% | 96.67±0.15% | 0.0922±0.0025 | 0.0535±0.0095 | 41.97% |

Prototype-direction consistency: `0.6490±0.0054`.  
Prototype membership-inference best balanced accuracy: `53.93±0.68%`.

## Environment

Reported experiments used Python 3.8, PyTorch 1.10, Ubuntu 22.04.4 LTS, NVIDIA RTX 4090D 24 GB, AMD EPYC 9654, and 755 GB RAM.

```bash
conda create -n fedcagc python=3.8 -y
conda activate fedcagc
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

Datasets are downloaded automatically by torchvision when needed.

## Main experiments

One CIFAR-10 FedCAGC run:

```bash
SEED=3047
CUDA_VISIBLE_DEVICES=0 python -u fedcagc_v2_all_methods.py \
  --dataset cifar10 --method fedcagc \
  --data_path ./data --device cuda \
  --num_clients 10 --rounds 100 --seed $SEED \
  --local_epochs 5 --local_bs 64 --local_lr 0.01 \
  --momentum 0.9 --weight_decay 1e-4 \
  --non_iid --gamma 0.5 \
  --watermark_source mnist --wm_train_size 100 --wm_test_size 200 --wm_beta 1.0 \
  --surgery_scope final_classifier --ema_rho 0.9 --tau_neg 0.0 \
  --wm_compensate --max_comp_scale 2.0 --grad_clip_norm 20 \
  --eval_every 5 --eval_tail 10 --snapshot_rounds 1,10,20,50,100 \
  --output_dir ./results/main/cifar10_fedcagc_$SEED
```

All main methods on both datasets and all three seeds:

```bash
bash scripts/run_main_3seeds.sh
```

## TraMark comparison

Run the corresponding FedCAGC main runs first so the same partition and watermark split can be reused:

```bash
bash scripts/run_tramark_3seeds.sh
```

## 30/50-client scalability

```bash
bash scripts/run_scalability_3seeds.sh
```

One 30-client FedCAGC run:

```bash
SEED=3047
CUDA_VISIBLE_DEVICES=0 python -u fedcagc_v2_scalability.py \
  --dataset cifar10 --method fedcagc \
  --data_path ./data --device cuda \
  --num_clients 30 --num_outputs 30 --rounds 100 --seed $SEED \
  --local_epochs 5 --local_bs 64 --local_lr 0.01 \
  --momentum 0.9 --weight_decay 1e-4 \
  --non_iid --gamma 0.5 \
  --watermark_source waffle --wm_train_size 100 --wm_test_size 200 \
  --ema_rho 0.9 --tau_neg 0.0 \
  --wm_compensate --max_comp_scale 2.0 --grad_clip_norm 20 \
  --output_dir ./results/experiments/scalability/c30_fedcagc_$SEED
```

## Mechanism experiments

The mechanism runner reproduces the same-round current-gradient reference, last-two-FC correction-space experiment, and PCGrad-history comparison:

```bash
bash scripts/run_mechanism_3seeds.sh
```

## Robustness experiments

Run 50-epoch clean fine-tuning and independent global unstructured magnitude pruning at `0,10,20,40,60,80,90,95,99%`:

```bash
bash scripts/run_robustness_3seeds.sh
```

## Analysis

The analysis script works on generated experiment artifacts:

```bash
python fedcagc_v2_analysis.py ownership --project .
python fedcagc_v2_analysis.py snapshot --project . --dataset cifar10 --seed 3047 --random_replays 20
python fedcagc_v2_analysis.py leakage --project . --dataset cifar10 --seed 3047
python fedcagc_v2_analysis.py stats --project .
```

Use `python <script>.py --help` for the full argument list.

## Output layout

```text
results/
├── main/
├── experiments/
└── analysis/
logs/
├── main/
└── experiments/
```

## References

1. McMahan et al. *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS, 2017.
2. Li et al. *FedIPR: Ownership Verification for Federated Deep Neural Network Models*. IEEE TPAMI, 2023.
3. Li et al. *Federated Learning Watermark Based on Model Backdoor*. Journal of Software, 2024.
4. Sun et al. *FedAWM: Adaptive Watermark Allocation in Non-IID Federated Learning*. Knowledge-Based Systems, 2026.
5. Xu et al. *Traceable Black-Box Watermarks for Federated Learning*. ICLR, 2026.
6. Yu et al. *Gradient Surgery for Multi-Task Learning*. NeurIPS, 2020.
