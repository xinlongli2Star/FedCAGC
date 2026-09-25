# FedCAGC：基于冲突感知梯度修正的联邦学习水印方法

本仓库提供论文《基于冲突感知梯度修正的联邦学习水印方法》的复现实验代码和运行说明。

FedCAGC 面向同步联邦学习中多客户端水印梯度的负向冲突问题，仅在最终分类层水印梯度子空间进行冲突检测与修正。服务器通过 EMA 维护各客户端历史水印梯度原型，模型聚合保持加权 FedAvg。

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

仓库仅包含代码和复现说明，运行结果写入 `results/` 和 `logs/`。

## 方法设定

- 仅在最终分类层水印梯度子空间进行冲突检测与修正；
- 服务器维护每个客户端的历史水印梯度原型；
- 历史水印梯度原型由修正前原始水印梯度通过 EMA 更新并进行 L2 归一化；
- 第 1 轮不执行冲突修正，仅完成联合训练并初始化原型；
- 第 2 轮起，根据当前水印梯度与其他客户端历史水印梯度原型的余弦相似度识别负向冲突；
- 按余弦相似度从小到大依次处理，并在每次投影后重新计算剩余方向的冲突关系；
- 只对仍满足负向冲突条件的方向继续修正；
- 投影后执行有界范数补偿，最大补偿系数为 2.0；
- 服务器聚合保持加权 FedAvg。

## 实验协议

| 参数 | 设置 |
|---|---|
| 主任务数据集 | FMNIST、CIFAR-10 |
| 客户端数 | 10 |
| 通信轮数 | 100 |
| 随机种子 | 3047、3048、3049 |
| 数据划分 | Dirichlet Non-IID，α=0.5 |
| Local Epoch | 5 |
| Batch Size | 64 |
| 优化器 | SGD |
| 学习率 | 0.01 |
| Momentum | 0.9 |
| Weight Decay | 1e-4 |
| 梯度裁剪 | 20.0 |
| 水印来源 | MNIST 官方数据集 |
| 水印训练集 | 每客户端100个，取自MNIST官方训练集 |
| 水印测试集 | 每客户端200个，取自MNIST官方测试集 |
| 水印训练/测试重叠 | 无 |

FedCAGC 默认参数：`rho=0.9`、`tau=0`、`smax=2.0`。

### 统计口径

- MTA、WMA、pre-WGC 和 post-WGC 采用随机种子 3047、3048、3049 的 mean±样本标准差。
- WGC 统一保留小数点后4位。
- WGC降低率由显示后的均值计算：`(mean(pre-WGC)-mean(post-WGC))/mean(pre-WGC)×100%`。
- FedAvg 不嵌入水印，因此水印相关指标记为“—”。
- TraMark 仅比较 MTA 和 WMA；其个性化水印区域不对应 FedCAGC 共享参数空间下的 WGC 定义。

## 论文参考结果

### 主实验

| 数据集 | 方法 | MTA | WMA | pre-WGC | post-WGC |
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

### 30/50客户端扩展实验

| 客户端数 | 方法 | MTA | WMA | pre-WGC | post-WGC | WGC降低率 |
|---:|---|---:|---:|---:|---:|---:|
| 30 | FedAvg | 85.12±0.63% | — | — | — | — |
| 30 | FedCAGC | 84.42±0.48% | 98.89±1.92% | 0.0317±0.0020 | 0.0194±0.0022 | 38.80% |
| 50 | FedAvg | 82.39±0.49% | — | — | — | — |
| 50 | FedCAGC | 79.78±0.38% | 69.38±6.91% | 0.0167±0.0007 | 0.0140±0.0018 | 16.17% |

### 微调与剪枝鲁棒性

| 数据集 | 初始WMA | 50 epoch微调 | 60%剪枝 | 80%剪枝 | 90%剪枝 | 95%剪枝 | 99%剪枝 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FMNIST | 92.60±1.30% | 85.45±5.71% | 91.85±1.26% | 87.10±1.78% | 66.97±8.16% | 36.70±4.80% | 14.55±3.48% |
| CIFAR-10 | 96.67±0.15% | 88.93±0.76% | 96.62±0.10% | 95.83±0.56% | 94.12±1.50% | 62.92±2.11% | 12.70±0.00% |


### 机制对比

| 设置 | MTA | WMA | pre-WGC | post-WGC | WGC降低率 |
|---|---:|---:|---:|---:|---:|
| EMA + 最终分类层 | 87.24±0.43% | 96.67±0.15% | 0.0922±0.0025 | 0.0535±0.0095 | 41.97% |
| 同轮真实水印梯度 + 最终分类层 | 87.16±0.33% | 96.77±0.24% | 0.0891±0.0083 | 0.0444±0.0086 | 50.17% |
| EMA + 最后两个FC层 | 87.47±0.31% | 96.78±0.48% | 0.0864±0.0087 | 0.0486±0.0018 | 43.75% |

### 投影顺序对比

| 投影顺序 | pre-WGC | post-WGC | WGC降低率 |
|---|---:|---:|---:|
| 余弦相似度升序 | 0.0883±0.0026 | 0.0535±0.0044 | 39.41% |
| 固定客户端编号顺序 | 0.0883±0.0026 | 0.0592±0.0057 | 32.96% |
| 随机顺序 | 0.0883±0.0026 | 0.0626±0.0056 | 29.11% |

### PCGrad-history对比

| 方法 | MTA | WMA | pre-WGC | post-WGC | WGC降低率 |
|---|---:|---:|---:|---:|---:|
| PCGrad-history | 87.30±0.36% | 94.40±0.74% | 0.0892±0.0023 | 0.0613±0.0119 | 31.28% |
| FedCAGC | 87.24±0.43% | 96.67±0.15% | 0.0922±0.0025 | 0.0535±0.0095 | 41.97% |

### 消融实验

| 方法 | MTA | WMA | pre-WGC | post-WGC | WGC降低率 |
|---|---:|---:|---:|---:|---:|
| FedIPR | 87.08±0.45% | 90.85±3.94% | 0.0846±0.0102 | 0.0846±0.0102 | 0.00% |
| FedCAGC w/o Comp | 87.36±0.45% | 96.60±0.53% | 0.0934±0.0059 | 0.0562±0.0079 | 39.83% |
| FedCAGC w/o EMA | 87.04±0.19% | 95.68±0.78% | 0.0896±0.0027 | 0.0670±0.0168 | 25.22% |
| FedCAGC | 87.24±0.43% | 96.67±0.15% | 0.0922±0.0025 | 0.0535±0.0095 | 41.97% |

历史原型与当前水印梯度的平均余弦相似度：`0.6490±0.0054`。  
成员推断实验最高平衡准确率：`53.93±0.68%`。

## 环境

论文实验环境：Python 3.8、PyTorch 1.10、Ubuntu 22.04.4 LTS、NVIDIA RTX 4090D 24 GB、AMD EPYC 9654、755 GB RAM。

```bash
conda create -n fedcagc python=3.8 -y
conda activate fedcagc
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

## 主实验

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

运行两数据集、五种主方法和三个随机种子：

```bash
bash scripts/run_main_3seeds.sh
```

## TraMark对比

先完成对应的FedCAGC主实验，以复用相同数据划分和水印样本：

```bash
bash scripts/run_tramark_3seeds.sh
```

## 30/50客户端扩展实验

```bash
bash scripts/run_scalability_3seeds.sh
```

单独运行30客户端FedCAGC：

```bash
SEED=3047
CUDA_VISIBLE_DEVICES=0 python -u fedcagc_v2_scalability.py \
  --dataset cifar10 --method fedcagc --data_path ./data --device cuda \
  --num_clients 30 --num_outputs 30 --rounds 100 --seed $SEED \
  --local_epochs 5 --local_bs 64 --local_lr 0.01 \
  --momentum 0.9 --weight_decay 1e-4 --non_iid --gamma 0.5 \
  --watermark_source waffle --wm_train_size 100 --wm_test_size 200 \
  --ema_rho 0.9 --tau_neg 0 --wm_compensate --max_comp_scale 2.0 \
  --grad_clip_norm 20 --output_dir ./results/experiments/scalability/c30_fedcagc_$SEED
```

## 机制实验

同轮真实水印梯度参考、最后两个FC层修正范围和PCGrad-history对比：

```bash
bash scripts/run_mechanism_3seeds.sh
```

## 鲁棒性实验

执行50轮无水印微调，以及`0,10,20,40,60,80,90,95,99%`全局非结构化幅值剪枝：

```bash
bash scripts/run_robustness_3seeds.sh
```

## 结果分析

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
