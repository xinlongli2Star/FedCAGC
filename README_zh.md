# FedCAGC：基于冲突感知梯度修正的联邦学习水印方法

本仓库提供论文《基于冲突感知梯度修正的联邦学习水印方法》的复现实验代码与运行说明。

[English README](README.md)

FedCAGC 面向同步联邦学习中多个客户端私有水印之间的负向梯度冲突问题，仅在最终分类层的水印梯度参数子空间内执行冲突检测与修正。服务器使用指数移动平均（EMA）为每个客户端维护历史水印梯度原型，服务器端模型聚合规则保持加权 FedAvg 不变。

## 文件结构

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

运行结果写入 `results/` 和 `logs/`。

## 方法设定

```text
第1轮
  主任务 + 水印联合训练
       ├── 本地模型 ------------------------------------> 加权 FedAvg
       └── 轮内平均原始分类层水印梯度 ------------------> 初始化历史原型

第2轮起
  全局模型 + 上一轮历史水印梯度原型集合
       │
       ▼
  当前原始水印梯度 g_i
       ├── 与其他客户端历史原型计算余弦相似度
       ├── 仅筛选负向冲突方向
       ├── 按余弦相似度从小到大排序
       ├── 投影移除负向冲突分量
       ├── 每次投影后重新判断剩余候选方向
       └── 有界范数补偿
       │
       ▼
  将修正后的分类层水印梯度写回完整水印梯度
       │
       ▼
  继续本地模型更新 ------------------------------------> 加权 FedAvg
                                                        └── EMA 更新历史原型
```

最终版本遵循以下规则：

- 仅移除负向冲突分量，不执行正向重叠消除；
- 冲突检测与修正仅作用于最终分类层；
- 第1轮不执行冲突修正，仅进行常规联合训练并初始化历史原型；
- 第2轮起开始执行冲突检测与修正；
- 历史水印梯度原型由未经修正的原始水印梯度构建；
- 不修改主任务梯度；
- 服务器端聚合规则保持加权 FedAvg。

## 实验协议

### 10客户端主实验

| 参数 | 设置 |
|---|---|
| 主任务数据集 | Fashion-MNIST（FMNIST）、CIFAR-10 |
| 模型 | FMNIST采用CNN；CIFAR-10采用AlexNet |
| 客户端数 | 10 |
| 通信轮数 | 100 |
| 随机种子 | 3047、3048、3049 |
| 数据划分 | Dirichlet Non-IID，α=0.5 |
| 本地训练轮数 | 5 |
| Batch Size | 64 |
| 优化器 | SGD |
| 学习率 | 0.01 |
| Momentum | 0.9 |
| Weight Decay | 1e-4 |
| 梯度裁剪 | 20.0 |
| 服务器聚合 | 加权 FedAvg |
| 水印来源 | MNIST官方数据集 |
| 水印训练集 | 每客户端100个样本，取自MNIST官方训练集 |
| 水印测试集 | 每客户端200个样本，取自MNIST官方测试集 |
| 水印训练/测试重叠 | 无 |
| 水印损失权重 β | 1.0 |

FedCAGC 默认参数：`rho=0.9`、`tau=0`、`smax=2.0`。

### 30/50客户端扩展实验

扩展实验采用与主实验不同的专门协议：

| 参数 | 设置 |
|---|---|
| 主任务数据集 | CIFAR-10 |
| 模型 | AlexNet |
| 客户端数 | 30或50 |
| 单客户端主任务样本数 | 固定为1000个不重复样本 |
| 主任务总样本数 | 30客户端使用30000个；50客户端使用50000个 |
| 数据划分 | 容量约束的类别级Dirichlet划分，α=0.5 |
| 水印 | 客户端专属WafflePattern |
| 水印训练/测试 | 每客户端100/200个 |
| 分类输出维度 | 与客户端数量一致 |
| 通信轮数 | 100 |
| 随机种子 | 3047、3048、3049 |
| 对比设置 | FedAvg、FedCAGC w/o Correction、FedCAGC |

### 统计口径

- MTA、WMA、WGC/pre-WGC 和 post-WGC 均采用随机种子 3047、3048、3049 的均值±样本标准差。
- WGC 统一保留小数点后4位。
- FedAvg 不嵌入水印，因此水印相关指标记为“—”。
- FedIPR、FLWB 和 FedAWM 不执行本文定义的冲突修正，因此其 WGC 表示最终分类层水印梯度的自然冲突水平，post-WGC 记为“—”。
- TraMark 仅比较 MTA 和 WMA；其个性化模型和独立水印区域不对应 FedCAGC 共享参数空间下的 WGC 定义。

## 论文参考结果

### 主对比实验

| 数据集 | 方法 | MTA | WMA | WGC / pre-WGC | post-WGC |
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

FedCAGC 在 FMNIST 上将 WGC 由 `0.0975±0.0040` 降至 `0.0648±0.0016`，降低 33.54%；在 CIFAR-10 上由 `0.0922±0.0025` 降至 `0.0535±0.0095`，降低 41.97%。

### 正确密钥与交叉身份验证

| 数据集 | WMA | Min-WMA | 交叉身份平均错误响应 | 交叉身份最大错误响应 |
|---|---:|---:|---:|---:|
| FMNIST | 92.60±1.30% | 85.33±5.75% | 1.137±0.143% | 4.33±1.26% |
| CIFAR-10 | 96.67±0.15% | 89.83±2.02% | 0.665±0.012% | 6.67±1.61% |

### 消融实验

| 方法 | MTA | WMA | WGC / pre-WGC | post-WGC |
|---|---:|---:|---:|---:|
| FedIPR | 87.08±0.45% | 90.85±3.94% | 0.0846±0.0102 | — |
| FedCAGC w/o Comp | 87.36±0.45% | 96.60±0.53% | 0.0934±0.0059 | 0.0562±0.0079 |
| FedCAGC w/o EMA | 87.04±0.19% | 95.68±0.78% | 0.0896±0.0027 | 0.0670±0.0168 |
| FedCAGC | 87.24±0.43% | 96.67±0.15% | 0.0922±0.0025 | 0.0535±0.0095 |

### 历史原型与修正范围分析

EMA 历史水印梯度原型与当前水印梯度之间的平均余弦相似度为：

`0.6490±0.0054`

| 设置 | MTA | WMA | pre-WGC | post-WGC | 修正维度 q | 原型存储 |
|---|---:|---:|---:|---:|---:|---:|
| EMA + 最终分类层 | 87.24±0.43% | 96.67±0.15% | 0.0922±0.0025 | 0.0535±0.0095 | 40,970 | 1.6388 MB |
| 同轮真实梯度 + 最终分类层 | 87.16±0.33% | 96.77±0.24% | 0.0891±0.0083 | 0.0444±0.0086 | 40,970 | 1.6388 MB |
| EMA + 最后两个FC层 | 87.47±0.31% | 96.78±0.48% | 0.0864±0.0087 | 0.0486±0.0018 | 16,822,282 | 672.89 MB |

“同轮真实梯度”设置仅作为理想参考，因为其需要集中共享客户端当前轮水印梯度信息，会增加客户端更新信息暴露风险。

### 投影顺序分析

| 投影顺序 | pre-WGC | post-WGC | WGC降低率 |
|---|---:|---:|---:|
| 余弦相似度升序 | 0.0883±0.0026 | 0.0535±0.0044 | 39.41% |
| 固定客户端编号顺序 | 0.0883±0.0026 | 0.0592±0.0057 | 32.96% |
| 随机顺序 | 0.0883±0.0026 | 0.0626±0.0056 | 29.11% |

### PCGrad-history机制对比

| 方法 | MTA | WMA | pre-WGC | post-WGC |
|---|---:|---:|---:|---:|
| PCGrad-history | 87.30±0.36% | 94.40±0.74% | 0.0892±0.0023 | 0.0613±0.0119 |
| FedCAGC | 87.24±0.43% | 96.67±0.15% | 0.0922±0.0025 | 0.0535±0.0095 |

对应的 WGC 降低率分别为 31.28% 和 41.97%。

### 关键超参数敏感性

| 参数 | 取值 | MTA | WMA | pre-WGC | post-WGC |
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

### 通信、存储与运行时间开销

| 数据集 | 完整模型参数大小 | 单客户端梯度/原型大小 | 额外上行总通信量/轮 | 额外下行总通信量/轮 | 服务器原型存储量 |
|---|---:|---:|---:|---:|---:|
| FMNIST | 6.653 MB | 0.0205 MB | 0.2052 MB | 2.052 MB | 0.2052 MB |
| CIFAR-10 | 143.421 MB | 0.1639 MB | 1.6388 MB | 16.388 MB | 1.6388 MB |

最终分类层梯度/原型大小约占完整模型参数大小的 0.31%（FMNIST）和 0.11%（CIFAR-10）。第2～5轮中，FedIPR 和 FedCAGC 的平均单轮运行时间分别为 `95.66±8.53 s` 和 `130.87±19.30 s`，FedCAGC 相对增加约 36.81%。

### 原型成员推断分析

三个随机种子下攻击者能够取得的最高平衡准确率（Balanced Accuracy，BA）为：

`53.93±0.68%`

攻击者根据候选样本水印梯度与目标客户端历史水印梯度原型之间的相似度进行成员判定，并遍历阈值取最高 BA。

### 微调与剪枝鲁棒性

| 数据集 | 初始WMA | 50轮无水印微调 | 60%剪枝 | 80%剪枝 | 90%剪枝 | 95%剪枝 | 99%剪枝 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FMNIST | 92.60±1.30% | 85.45±5.71% | 91.85±1.26% | 87.10±1.78% | 66.97±8.16% | 36.70±4.80% | 14.55±3.48% |
| CIFAR-10 | 96.67±0.15% | 88.93±0.76% | 96.62±0.10% | 95.83±0.56% | 94.12±1.50% | 62.92±2.11% | 12.70±0.00% |

### 30/50客户端扩展实验

| 客户端数 | 方法 | MTA | WMA | WGC / pre-WGC | post-WGC |
|---:|---|---:|---:|---:|---:|
| 30 | FedAvg | 79.85±0.86% | — | — | — |
| 30 | FedCAGC w/o Correction | 73.84±1.47% | 77.74±10.13% | 0.0289±0.0026 | — |
| 30 | FedCAGC | 75.35±1.75% | 98.17±1.62% | 0.0292±0.0031 | 0.0214±0.0023 |
| 50 | FedAvg | 81.18±0.39% | — | — | — |
| 50 | FedCAGC w/o Correction | 66.50±9.07% | 39.53±8.15% | 0.0162±0.0006 | — |
| 50 | FedCAGC | 75.90±0.74% | 72.81±4.28% | 0.0160±0.0013 | 0.0148±0.0010 |

## 实验环境

论文实验环境：Python 3.8、PyTorch 1.10、Ubuntu 22.04.4 LTS、NVIDIA RTX 4090D 24 GB、AMD EPYC 9654、755 GB RAM。

```bash
conda create -n fedcagc python=3.8 -y
conda activate fedcagc
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

数据集由 torchvision 在需要时自动下载。

## 复现实验

### 主实验

单独运行 CIFAR-10 FedCAGC：

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
  --surgery_scope final_classifier --ema_rho 0.9 --tau_neg 0 \
  --wm_compensate --max_comp_scale 2.0 --grad_clip_norm 20 \
  --eval_every 5 --eval_tail 10 --snapshot_rounds 1,10,20,50,100 \
  --output_dir ./results/main/cifar10_fedcagc_$SEED
```

运行两个数据集、全部主对比方法和3个随机种子：

```bash
bash scripts/run_main_3seeds.sh
```

### TraMark对比

```bash
bash scripts/run_tramark_3seeds.sh
```

### 30/50客户端扩展实验

```bash
bash scripts/run_scalability_3seeds.sh
```

单独运行30客户端FedCAGC：

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
  --surgery_scope final_classifier --ema_rho 0.9 --tau_neg 0 \
  --wm_compensate --max_comp_scale 2.0 --grad_clip_norm 20 \
  --output_dir ./results/experiments/scalability/c30_fedcagc_$SEED
```

若需复现表9中的无冲突修正设置，将 `--method fedcagc` 替换为：

```bash
--method fedcagc_nocorr
```

### 机制验证实验

```bash
bash scripts/run_mechanism_3seeds.sh
```

该入口覆盖正文中的同轮真实梯度理想参考、最后两个FC层修正范围、投影顺序、PCGrad-history及关键机制相关实验。

### 鲁棒性实验

```bash
bash scripts/run_robustness_3seeds.sh
```

正文报告50轮无水印微调，以及60%、80%、90%、95%和99%的全局非结构化幅值剪枝结果；运行脚本可额外计算中间剪枝比例以生成完整曲线。

### 结果分析

```bash
python fedcagc_v2_analysis.py ownership --project .
python fedcagc_v2_analysis.py snapshot --project . --dataset cifar10 --seed 3047 --random_replays 20
python fedcagc_v2_analysis.py leakage --project . --dataset cifar10 --seed 3047
python fedcagc_v2_analysis.py stats --project .
```

完整参数可通过 `python <script>.py --help` 查看。

## 输出目录

```text
results/
├── main/
├── experiments/
└── analysis/
logs/
├── main/
└── experiments/
```

## 参考文献

1. McMahan et al. *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS, 2017.
2. Li et al. *FedIPR: Ownership Verification for Federated Deep Neural Network Models*. IEEE TPAMI, 2023.
3. Li et al. *Federated Learning Watermark Based on Model Backdoor*. Journal of Software, 2024.
4. Sun et al. *FedAWM: Adaptive Watermark Allocation in Non-IID Federated Learning*. Knowledge-Based Systems, 2025.
5. Xu et al. *Traceable Black-Box Watermarks for Federated Learning*. ICLR, 2026.
6. Yu et al. *Gradient Surgery for Multi-Task Learning*. NeurIPS, 2020.
7. Liu et al. *Conflict-Averse Gradient Descent for Multi-task Learning*. NeurIPS, 2021.
