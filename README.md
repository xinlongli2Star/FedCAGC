# FedCAGC: Conflict-Aware Gradient Correction for Federated Learning Watermarking

Official reproduction code for **“基于冲突感知梯度修正的联邦学习水印方法”**.

[中文说明](README_zh.md)

FedCAGC targets negative gradient interference among multiple client-specific private watermarks in synchronous federated learning. Conflict detection and correction are restricted to the watermark-gradient subspace of the final classification layer. The server maintains one historical watermark-gradient prototype for each client using an exponential moving average (EMA), while the server-side model aggregation rule remains weighted FedAvg.

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

Runtime outputs are written to `results/` and `logs/`.

## Method overview

```text
Round 1
  local main-task + watermark training
       ├── local model --------------------------------> weighted FedAvg
       └── mean raw final-classifier watermark gradient
                                                      -> initialize client prototypes

Round 2+
  global model + previous-round historical watermark-gradient prototypes
       │
       ▼
  current raw watermark gradient g_i
       ├── compute cosine similarity with other clients' prototypes
       ├── select only negative-conflict directions
       ├── sort candidates by ascending cosine similarity
       ├── project the negative component
       ├── re-check the remaining candidates after every projection
       └── bounded norm compensation
       │
       ▼
  write corrected final-classifier watermark gradient back
       │
       ▼
  continue local optimization ------------------------> weighted FedAvg
                                                        └── EMA prototype update
```

The finalized implementation follows these rules:

- only negative-conflict components are removed;
- no positive-overlap removal is performed;
- correction is limited to the final classification layer;
- round 1 performs no correction and is used to initialize historical prototypes;
- correction starts from round 2;
- historical prototypes are built from uncorrected raw watermark gradients;
- the main-task gradient is not modified;
- server aggregation remains weighted FedAvg.

## Experimental protocol

### Main 10-client experiments

| Parameter | Setting |
|---|---|
| Main datasets | Fashion-MNIST (FMNIST), CIFAR-10 |
| Model | CNN for FMNIST; AlexNet for CIFAR-10 |
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
| WM train | 100 samples/client from the official MNIST training split |
| WM test | 200 samples/client from the official MNIST test split |
| WM train/test overlap | none |
| Watermark loss weight β | 1.0 |

Default FedCAGC parameters: `rho=0.9`, `tau=0`, `smax=2.0`.

### 30/50-client scalability protocol

The scalability experiment uses a separate paper-aligned protocol:

| Parameter | Setting |
|---|---|
| Main dataset | CIFAR-10 |
| Model | AlexNet |
| Clients | 30 or 50 |
| Main-task samples/client | exactly 1000 unique samples |
| Total main-task samples | 30,000 for 30 clients; 50,000 for 50 clients |
| Partition | capacity-constrained class-wise Dirichlet, α=0.5 |
| Watermark | client-specific WafflePattern |
| WM train/test | 100 / 200 samples per client |
| Output dimension | equal to client count |
| Communication rounds | 100 |
| Seeds | 3047, 3048, 3049 |
| Compared settings | FedAvg, FedCAGC w/o Correction, FedCAGC |

### Reporting convention

- MTA, WMA, WGC/pre-WGC and post-WGC are reported as mean ± sample standard deviation over seeds 3047/3048/3049.
- WGC values are displayed to four decimal places.
- FedAvg contains no watermark, so watermark-related metrics are shown as `—`.
- For FedIPR, FLWB, and FedAWM, the reported WGC is the natural conflict level of the final-classifier watermark gradients; these methods do not perform the FedCAGC correction, so post-WGC is shown as `—`.
- TraMark is compared using MTA and WMA only because its personalized watermark regions do not correspond to the shared-parameter WGC definition used by FedCAGC.

## Paper reference results

### Main comparison

| Dataset | Method | MTA | WMA | WGC / pre-WGC | post-WGC |
|---|---|---:|---:|---:|---:|
| FMNIST | FedAvg | 91.85±0.15% | — | — | — |
| FMNIST | FedIPR | 91.69±0.10% | 89.58±1.44% | 0.0984±0.0064 | — |
| FMNIST | FLWB | 91.82±0.23% | 78.48±2.71% | 0.0956±0.0065 | — |
| FMNIST | FedAWM | 91.62±0.22% | 89.83±1.55% | 0.0959±0.0051 | — |
| FMNIST | TraMark | 90.29±0.28% | 92.50±0.36% | — | — |
| FMNIST | FedCAGC | 91.69±0.05% | 92.60±1.30% | 0.0975±0.0040 | 0.0648±0.0016 |
| CIFAR-10 | FedAvg | 87.40±0.43% | — | — | — |
| CIFAR-10 | FedIPR | 87.08±0.45% | 90.85±3.94% | 0.0846±0.0102 | — |
| CIFAR-10 | FLWB | 87.24±0.27% | 76.68±3.75% | 0.1003±0.0099 | — |
| CIFAR-10 | FedAWM | 87.42±0.17% | 94.20±0.56% | 0.0928±0.0040 | — |
| CIFAR-10 | TraMark | 85.97±0.37% | 83.60±2.73% | — | — |
| CIFAR-10 | FedCAGC | 87.24±0.43% | 96.67±0.15% | 0.0922±0.0025 | 0.0535±0.0095 |

FedCAGC reduces WGC from `0.0975±0.0040` to `0.0648±0.0016` on FMNIST and from `0.0922±0.0025` to `0.0535±0.0095` on CIFAR-10, corresponding to reductions of 33.54% and 41.97%, respectively.

### Correct-key and cross-identity verification

| Dataset | WMA | Min-WMA | Mean cross-identity error response | Maximum cross-identity error response |
|---|---:|---:|---:|---:|
| FMNIST | 92.60±1.30% | 85.33±5.75% | 1.137±0.143% | 4.33±1.26% |
| CIFAR-10 | 96.67±0.15% | 89.83±2.02% | 0.665±0.012% | 6.67±1.61% |

### Ablation

| Method | MTA | WMA | WGC / pre-WGC | post-WGC |
|---|---:|---:|---:|---:|
| FedIPR | 87.08±0.45% | 90.85±3.94% | 0.0846±0.0102 | — |
| FedCAGC w/o Comp | 87.36±0.45% | 96.60±0.53% | 0.0934±0.0059 | 0.0562±0.0079 |
| FedCAGC w/o EMA | 87.04±0.19% | 95.68±0.78% | 0.0896±0.0027 | 0.0670±0.0168 |
| FedCAGC | 87.24±0.43% | 96.67±0.15% | 0.0922±0.0025 | 0.0535±0.0095 |

### Historical prototype and correction-scope analysis

Prototype-direction consistency (mean cosine similarity between the EMA historical prototype and the current watermark gradient):

`0.6490±0.0054`

| Setting | MTA | WMA | pre-WGC | post-WGC | Correction dimension q | Prototype storage |
|---|---:|---:|---:|---:|---:|---:|
| EMA + final classifier | 87.24±0.43% | 96.67±0.15% | 0.0922±0.0025 | 0.0535±0.0095 | 40,970 | 1.6388 MB |
| Same-round true gradient + final classifier | 87.16±0.33% | 96.77±0.24% | 0.0891±0.0083 | 0.0444±0.0086 | 40,970 | 1.6388 MB |
| EMA + last two FC layers | 87.47±0.31% | 96.78±0.48% | 0.0864±0.0087 | 0.0486±0.0018 | 16,822,282 | 672.89 MB |

The same-round true-gradient setting is used only as an idealized reference because it requires centralized sharing of current-round client watermark-gradient information.

### Projection-order analysis

| Projection order | pre-WGC | post-WGC | WGC reduction |
|---|---:|---:|---:|
| Cosine similarity ascending | 0.0883±0.0026 | 0.0535±0.0044 | 39.41% |
| Fixed client-ID order | 0.0883±0.0026 | 0.0592±0.0057 | 32.96% |
| Random order | 0.0883±0.0026 | 0.0626±0.0056 | 29.11% |

### PCGrad-history comparison

| Method | MTA | WMA | pre-WGC | post-WGC |
|---|---:|---:|---:|---:|
| PCGrad-history | 87.30±0.36% | 94.40±0.74% | 0.0892±0.0023 | 0.0613±0.0119 |
| FedCAGC | 87.24±0.43% | 96.67±0.15% | 0.0922±0.0025 | 0.0535±0.0095 |

The corresponding WGC reductions are 31.28% for PCGrad-history and 41.97% for FedCAGC.

### Hyperparameter sensitivity

| Parameter | Value | MTA | WMA | pre-WGC | post-WGC |
|---|---:|---:|---:|---:|---:|
| ρ | 0.8 | 87.37±0.49% | 95.92±1.37% | 0.0927±0.0076 | 0.0520±0.0073 |
| ρ | 0.9 | 87.24±0.43% | 96.67±0.15% | 0.0922±0.0025 | 0.0535±0.0095 |
| ρ | 0.99 | 87.47±0.09% | 96.88±1.11% | 0.0947±0.0018 | 0.0603±0.0052 |
| τ | 0 | 87.24±0.43% | 96.67±0.15% | 0.0922±0.0025 | 0.0535±0.0095 |
| τ | -0.05 | 87.28±0.33% | 96.88±0.08% | 0.0922±0.0037 | 0.0640±0.0077 |
| τ | -0.10 | 87.31±0.11% | 93.82±5.48% | 0.0909±0.0065 | 0.0710±0.0023 |
| smax | 1.5 | 87.31±0.43% | 96.78±0.21% | 0.0914±0.0049 | 0.0540±0.0061 |
| smax | 2.0 | 87.24±0.43% | 96.67±0.15% | 0.0922±0.0025 | 0.0535±0.0095 |
| smax | 2.5 | 87.33±0.23% | 96.18±1.18% | 0.0875±0.0019 | 0.0534±0.0039 |

### Communication, storage, and runtime overhead

| Dataset | Full model size | Gradient/prototype per client | Extra uplink/round | Extra downlink/round | Server prototype storage |
|---|---:|---:|---:|---:|---:|
| FMNIST | 6.653 MB | 0.0205 MB | 0.2052 MB | 2.052 MB | 0.2052 MB |
| CIFAR-10 | 143.421 MB | 0.1639 MB | 1.6388 MB | 16.388 MB | 1.6388 MB |

The final-classifier gradient/prototype occupies approximately 0.31% of the full FMNIST model and 0.11% of the full CIFAR-10 model. In rounds 2–5, the average per-round runtime is `95.66±8.53 s` for FedIPR and `130.87±19.30 s` for FedCAGC, corresponding to an increase of approximately 36.81%.

### Prototype membership-inference analysis

Best balanced accuracy over the three seeds:

`53.93±0.68%`

The attack compares candidate-sample watermark gradients with the target client's historical watermark-gradient prototype and selects the threshold that maximizes balanced accuracy.

### Fine-tuning and pruning robustness

| Dataset | Initial WMA | 50-epoch clean fine-tuning | 60% pruning | 80% pruning | 90% pruning | 95% pruning | 99% pruning |
|---|---:|---:|---:|---:|---:|---:|---:|
| FMNIST | 92.60±1.30% | 85.45±5.71% | 91.85±1.26% | 87.10±1.78% | 66.97±8.16% | 36.70±4.80% | 14.55±3.48% |
| CIFAR-10 | 96.67±0.15% | 88.93±0.76% | 96.62±0.10% | 95.83±0.56% | 94.12±1.50% | 62.92±2.11% | 12.70±0.00% |

### 30/50-client scalability

| Clients | Method | MTA | WMA | WGC / pre-WGC | post-WGC |
|---:|---|---:|---:|---:|---:|
| 30 | FedAvg | 79.85±0.86% | — | — | — |
| 30 | FedCAGC w/o Correction | 73.84±1.47% | 77.74±10.13% | 0.0289±0.0026 | — |
| 30 | FedCAGC | 75.35±1.75% | 98.17±1.62% | 0.0292±0.0031 | 0.0214±0.0023 |
| 50 | FedAvg | 81.18±0.39% | — | — | — |
| 50 | FedCAGC w/o Correction | 66.50±9.07% | 39.53±8.15% | 0.0162±0.0006 | — |
| 50 | FedCAGC | 75.90±0.74% | 72.81±4.28% | 0.0160±0.0013 | 0.0148±0.0010 |

## Environment

Reported experiments used Python 3.8, PyTorch 1.10, Ubuntu 22.04.4 LTS, NVIDIA RTX 4090D 24 GB, AMD EPYC 9654, and 755 GB RAM.

```bash
conda create -n fedcagc python=3.8 -y
conda activate fedcagc
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

Datasets are downloaded automatically by torchvision when needed.

## Reproduction

### Main experiments

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

### TraMark comparison

```bash
bash scripts/run_tramark_3seeds.sh
```

### 30/50-client scalability

```bash
bash scripts/run_scalability_3seeds.sh
```

One 30-client FedCAGC run:

```bash
SEED=3047
CUDA_VISIBLE_DEVICES=0 python -u fedcagc_v2_scalability.py \
  --dataset cifar10 --method fedcagc \
  --data_path ./data --device cuda \
  --num_clients 30 --num_outputs 30 --samples_per_client 1000 \
  --rounds 100 --seed $SEED \
  --local_epochs 5 --local_bs 64 --local_lr 0.01 \
  --momentum 0.9 --weight_decay 1e-4 \
  --non_iid --gamma 0.5 \
  --watermark_source waffle --wm_train_size 100 --wm_test_size 200 --wm_beta 1.0 \
  --surgery_scope final_classifier --ema_rho 0.9 --tau_neg 0.0 \
  --wm_compensate --max_comp_scale 2.0 --grad_clip_norm 20 \
  --output_dir ./results/experiments/scalability/c30_fedcagc_$SEED
```

To reproduce the ablation baseline used in the scalability table, replace `--method fedcagc` with:

```bash
--method fedcagc_nocorr
```

### Mechanism experiments

```bash
bash scripts/run_mechanism_3seeds.sh
```

This runner covers the same-round current-gradient reference, last-two-FC correction range, projection-order analysis, PCGrad-history comparison, and key mechanism settings used in the paper.

### Robustness experiments

```bash
bash scripts/run_robustness_3seeds.sh
```

The paper reports 50 epochs of clean fine-tuning and global unstructured magnitude pruning at 60%, 80%, 90%, 95%, and 99%. The robustness runner may additionally evaluate intermediate pruning ratios for curve generation.

### Analysis

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
4. Sun et al. *FedAWM: Adaptive Watermark Allocation in Non-IID Federated Learning*. Knowledge-Based Systems, 2025.
5. Xu et al. *Traceable Black-Box Watermarks for Federated Learning*. ICLR, 2026.
6. Yu et al. *Gradient Surgery for Multi-Task Learning*. NeurIPS, 2020.
7. Liu et al. *Conflict-Averse Gradient Descent for Multi-task Learning*. NeurIPS, 2021.
