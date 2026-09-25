"""FedCAGC V2 experiment runner.

Methods included:
  fedavg   : no-watermark reference.
  fedipr   : FedIPR black-box/backdoor branch under the unified watermark protocol.
  flwb     : FLWB stepwise/separated watermark training under the unified protocol.
  fedawm   : project reproduction of FedAWM under the unified protocol.
  fedcagc  : conflict-aware gradient correction (FedCAGC).

Formal V2 common setting:
  10 clients, Dirichlet alpha=0.5, 5 local epochs, batch size 64, SGD lr=0.01,
  100 communication rounds, seeds 3047/3048/3049.

Watermark protocol:
  FedIPR / FLWB / FedAWM / FedCAGC use the same watermark train/test split for
  controlled comparison: 100 samples/client from official MNIST train and
  200 samples/client from official MNIST test.
"""

import argparse
import hashlib
import copy
import csv
import json
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, transforms


@dataclass
class DatasetSpec:
    name: str
    default_rounds: int
    image_size: int
    channels: int
    num_classes: int
    mean: Sequence[float]
    std: Sequence[float]


DATASET_SPECS = {
    "fmnist": DatasetSpec("fmnist", 100, 28, 1, 10, (0.5,), (0.5,)),
    "cifar10": DatasetSpec("cifar10", 100, 32, 3, 10, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": DatasetSpec("cifar100", 100, 32, 3, 100, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


class CNN4Fmnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class VGG16(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
               512, 512, 512, "M", 512, 512, 512, "M"]
        layers = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend([
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                ])
                in_channels = v
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class WafflePatternDataset(Dataset):
    def __init__(self, client_id: int, target_label: int, size: int, spec: DatasetSpec,
                 seed: int, index_offset: int = 0):
        self.client_id = client_id
        self.target_label = target_label
        self.size = size
        self.spec = spec
        self.seed = seed
        self.index_offset = index_offset
        self.mean = torch.tensor(spec.mean, dtype=torch.float32).view(spec.channels, 1, 1)
        self.std = torch.tensor(spec.std, dtype=torch.float32).view(spec.channels, 1, 1)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        index = self.index_offset + index
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.client_id * 1000003 + index)
        h = self.spec.image_size
        w = self.spec.image_size
        c = self.spec.channels
        base = 0.15 + 0.20 * torch.rand((c, h, w), generator=generator)
        row_gap = 3 + (self.client_id % 5)
        col_gap = 4 + ((self.client_id // 5) % 6)
        row_offset = (self.client_id * 2) % row_gap
        col_offset = (self.client_id * 3) % col_gap
        base[:, row_offset::row_gap, :] = 0.85
        base[:, :, col_offset::col_gap] = 0.85
        patch_size = 3 + (self.client_id % 3)
        top = (self.client_id * 7) % max(1, h - patch_size)
        left = (self.client_id * 11) % max(1, w - patch_size)
        base[:, top:top + patch_size, left:left + patch_size] = 1.0
        if c == 3:
            color = torch.tensor([
                ((self.client_id * 37) % 255) / 255.0,
                ((self.client_id * 67 + 41) % 255) / 255.0,
                ((self.client_id * 97 + 83) % 255) / 255.0,
            ], dtype=torch.float32).view(3, 1, 1)
            base = 0.55 * base + 0.45 * color
        base = torch.clamp(base, 0.0, 1.0)
        x = (base - self.mean) / self.std
        return x, int(self.target_label)


def parse_args():
    parser = argparse.ArgumentParser(
        description="FedCAGC V2 unified runner: FedIPR/FLWB/FedAWM/FedCAGC."
    )
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["fmnist", "cifar10"])
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--method", type=str, default="fedcagc",
                        choices=["fedavg", "fedipr", "flwb", "fedawm", "fedcagc"])
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--num_outputs", type=int, default=0,
                        help="0 uses the main dataset class number. Use 30 or 50 for CIFAR-10 scale experiments.")
    parser.add_argument("--rounds", type=int, default=0, help="0 uses dataset default rounds.")
    parser.add_argument("--seed", type=int, default=3047)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--fast_tf32", action="store_true",
                        help="Enable TF32 matmul/cuDNN on NVIDIA Ampere/Ada GPUs. Faster, but numerics may differ slightly.")
    parser.add_argument("--profile_round", action="store_true",
                        help="Print per-round train/aggregate/eval timing breakdown.")

    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--local_bs", type=int, default=64)
    parser.add_argument("--local_lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)

    partition_group = parser.add_mutually_exclusive_group()
    partition_group.add_argument("--non_iid", dest="non_iid", action="store_true")
    partition_group.add_argument("--iid", dest="non_iid", action="store_false")
    parser.set_defaults(non_iid=True)
    parser.add_argument("--gamma", type=float, default=0.5)

    parser.add_argument("--surgery_scope", type=str, default="final_classifier",
                        choices=["final_classifier"],
                        help="FedCAGC V2 is fixed to the final classification layer.")
    parser.add_argument("--ema_rho", type=float, default=0.9)
    parser.add_argument("--tau_neg", type=float, default=0.0)
    parser.add_argument("--surgery_max_clients", type=int, default=0,
                        help="0 processes every negative-conflict prototype (the finalized setting).")
    parser.add_argument("--wm_beta", type=float, default=1.0)
    parser.add_argument("--watermark_source", type=str, default="mnist", choices=["mnist", "waffle"])
    parser.add_argument("--wm_train_size", type=int, default=100)
    parser.add_argument("--wm_test_size", type=int, default=200,
                        help="Watermark test samples per client. V2 fixes this to 200 from the method-appropriate official test split.")
    compensate_group = parser.add_mutually_exclusive_group()
    compensate_group.add_argument("--wm_compensate", dest="wm_compensate", action="store_true")
    compensate_group.add_argument("--no_wm_compensate", dest="wm_compensate", action="store_false")
    parser.set_defaults(wm_compensate=True)
    parser.add_argument("--max_comp_scale", type=float, default=2.0)
    parser.add_argument("--grad_clip_norm", type=float, default=20.0)

    # FLWB: stepwise watermark update (Algorithm 5 in Li et al., Journal of Software 2024).
    # The source paper defines an enhancement coefficient lambda but does not give a unique
    # universal value in the experimental table; keep the established project setting (1.0)
    # and expose it explicitly for reproducibility.
    parser.add_argument("--flwb_lambda", type=float, default=1.0)
    parser.add_argument("--flwb_wm_steps", type=int, default=1)

    # FedAWM: preserve the established project reproduction used in the unified comparison protocol.
    # Allocation is estimated from WATERMARK-TRAINING data only; the official test split is never
    # used for allocation, preventing test leakage.
    parser.add_argument("--awm_temperature", type=float, default=1.0)
    parser.add_argument("--awm_min_scale", type=float, default=0.5)
    parser.add_argument("--awm_max_scale", type=float, default=2.0)
    parser.add_argument("--awm_ema", type=float, default=0.8)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--eval_tail", type=int, default=10,
                        help="Always evaluate every round in the final N rounds; 0 disables.")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Engineering-only short run for code checks.")
    parser.add_argument("--smoke_samples_per_client", type=int, default=128)
    parser.add_argument("--smoke_test_samples", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="./results/v2")
    parser.add_argument("--csv_path", type=str, default="",
                        help="Optional CSV path; defaults to OUTPUT_DIR/round_metrics.csv.")
    parser.add_argument("--snapshot_rounds", type=str, default="1,10,20,50,100",
                        help="Comma-separated rounds at which normalized prototypes are saved.")

    args = parser.parse_args()
    if args.watermark_source == "mnist" and args.num_clients > 10:
        parser.error("MNIST-label watermark supports at most 10 clients.")
    if not 0.0 <= args.ema_rho < 1.0:
        parser.error("--ema_rho must be in [0, 1).")
    if args.tau_neg < 0:
        parser.error("--tau_neg must be non-negative.")
    if args.surgery_max_clients < 0:
        parser.error("--surgery_max_clients must be non-negative.")
    if args.gamma <= 0:
        parser.error("--gamma must be positive.")
    if args.flwb_lambda <= 0 or args.flwb_wm_steps <= 0:
        parser.error("FLWB parameters must be positive.")
    if args.awm_temperature <= 0:
        parser.error("--awm_temperature must be positive.")
    if not 0 < args.awm_min_scale <= args.awm_max_scale:
        parser.error("Require 0 < --awm_min_scale <= --awm_max_scale.")
    if not 0 <= args.awm_ema < 1:
        parser.error("--awm_ema must be in [0,1).")
    if args.wm_train_size <= 0:
        parser.error("--wm_train_size must be positive.")
    if args.wm_test_size < 0:
        parser.error("--wm_test_size must be non-negative.")
    if args.wm_train_size != 100 or args.wm_test_size != 200:
        parser.error("V2 fixes --wm_train_size=100 and --wm_test_size=200.")
    if args.eval_every <= 0:
        parser.error("--eval_every must be positive.")
    if args.eval_tail < 0:
        parser.error("--eval_tail must be non-negative.")
    if args.smoke_samples_per_client <= 0 or args.smoke_test_samples <= 0:
        parser.error("Smoke-test sample counts must be positive.")
    if args.num_outputs < 0:
        parser.error("--num_outputs must be non-negative.")
    if args.watermark_source == "mnist" and args.num_outputs not in [0, 10]:
        parser.error("MNIST-label watermark is intended for 10-output experiments. Use --watermark_source waffle for expanded outputs.")
    if args.watermark_source == "waffle" and args.num_outputs > 0 and args.num_clients > args.num_outputs:
        parser.error("--num_clients must not exceed --num_outputs for WafflePattern watermarking.")
    try:
        args.snapshot_rounds = sorted({
            int(item.strip()) for item in args.snapshot_rounds.split(",") if item.strip()
        })
    except ValueError:
        parser.error("--snapshot_rounds must be a comma-separated list of integers.")
    if any(round_id <= 0 for round_id in args.snapshot_rounds):
        parser.error("--snapshot_rounds must contain positive round numbers.")

    spec = DATASET_SPECS[args.dataset]
    if args.num_outputs <= 0:
        args.num_outputs = spec.num_classes
    if args.rounds <= 0:
        args.rounds = spec.default_rounds
    args.snapshot_rounds = sorted({r for r in args.snapshot_rounds if r <= args.rounds} | {args.rounds})

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = torch.device(args.device)
    args.output_dir = str(Path(args.output_dir).expanduser().resolve())
    if not args.csv_path:
        args.csv_path = str(Path(args.output_dir) / "round_metrics.csv")
    else:
        args.csv_path = str(Path(args.csv_path).expanduser().resolve())
    return args



def fix_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 固定随机种子，但不强制使用确定性算法，避免明显降低GPU训练速度
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_model(dataset: str, num_outputs: int) -> nn.Module:
    if dataset == "fmnist":
        return CNN4Fmnist()
    if dataset == "cifar10":
        return AlexNet(num_classes=num_outputs)
    if dataset == "cifar100":
        return VGG16(num_classes=num_outputs)
    raise ValueError(dataset)


def get_main_datasets(args):
    spec = DATASET_SPECS[args.dataset]
    normalize = transforms.Normalize(spec.mean, spec.std)
    if args.dataset == "fmnist":
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        train_dataset = datasets.FashionMNIST(args.data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(args.data_path, train=False, download=True, transform=transform)
    elif args.dataset in ["cifar10", "cifar100"]:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        dataset_cls = datasets.CIFAR10 if args.dataset == "cifar10" else datasets.CIFAR100
        train_dataset = dataset_cls(args.data_path, train=True, download=True, transform=transform_train)
        test_dataset = dataset_cls(args.data_path, train=False, download=True, transform=transform_test)
    else:
        raise ValueError(args.dataset)

    train_dataset.targets = torch.as_tensor(train_dataset.targets, dtype=torch.long)
    test_dataset.targets = torch.as_tensor(test_dataset.targets, dtype=torch.long)
    return train_dataset, test_dataset


def get_mnist_watermark_transform(args):
    spec = DATASET_SPECS[args.dataset]
    normalize = transforms.Normalize(spec.mean, spec.std)
    steps = []
    if spec.image_size != 28:
        steps.append(transforms.Resize((spec.image_size, spec.image_size)))
    if spec.channels == 3:
        steps.append(transforms.Grayscale(num_output_channels=3))
    steps.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(steps)


def build_watermark_datasets(args):
    """Build the V2 watermark datasets with official MNIST splits.

    Formal V2 protocol for the MNIST watermark source:
      - 100 watermark-training samples/client are sampled only from the official
        MNIST training split (train=True).
      - 200 watermark-test samples/client are sampled only from the official
        MNIST test split (train=False).
      - The two sets are therefore disjoint by the official dataset partition.

    The returned wm_extra_sets is retained only for backward compatibility with
    older result fields; in V2 it aliases the official watermark test set and is
    never concatenated with the watermark training set.
    """
    spec = DATASET_SPECS[args.dataset]
    target_labels = list(range(args.num_clients))

    if args.watermark_source == "waffle":
        # Synthetic WafflePattern has no official train/test split. Keep strictly
        # disjoint index ranges for optional non-formal source-generalization tests.
        wm_train_sets, wm_extra_sets, wm_test_sets = [], [], []
        metadata = {
            "source": "waffle",
            "split_protocol": "synthetic_disjoint_train100_test200",
            "formal_v2_protocol": False,
            "clients": {},
        }
        for cid, target_label in enumerate(target_labels):
            train_set = WafflePatternDataset(
                cid, target_label, args.wm_train_size, spec, args.seed, index_offset=0
            )
            test_set = WafflePatternDataset(
                cid, target_label, args.wm_test_size, spec, args.seed,
                index_offset=args.wm_train_size
            )
            wm_train_sets.append(train_set)
            wm_extra_sets.append(test_set)
            wm_test_sets.append(test_set)
            metadata["clients"][str(cid)] = {
                "target_label": int(target_label),
                "train_source_split": "synthetic",
                "test_source_split": "synthetic",
                "train_pattern_indices": list(range(0, args.wm_train_size)),
                "test_pattern_indices": list(
                    range(args.wm_train_size, args.wm_train_size + args.wm_test_size)
                ),
                "train_test_overlap": 0,
            }
        return wm_train_sets, wm_extra_sets, wm_test_sets, target_labels, metadata

    transform = get_mnist_watermark_transform(args)

    # Formal V2: training watermarks come from the official MNIST training split,
    # while verification watermarks come from the official MNIST test split.
    train_mnist = datasets.MNIST(
        args.data_path, train=True, download=True, transform=transform
    )
    test_mnist = datasets.MNIST(
        args.data_path, train=False, download=True, transform=transform
    )
    train_targets = np.asarray(train_mnist.targets)
    test_targets = np.asarray(test_mnist.targets)

    wm_train_sets, wm_extra_sets, wm_test_sets = [], [], []
    metadata = {
        "source": "mnist",
        "split_protocol": "official_mnist_train100_test200",
        "formal_v2_protocol": True,
        "train_source_split": "MNIST official train",
        "test_source_split": "MNIST official test",
        "clients": {},
    }

    for label in target_labels:
        train_candidates = np.where(train_targets == label)[0]
        test_candidates = np.where(test_targets == label)[0]

        # Keep the train-side sampling rule stable across runs for the same seed.
        train_rng = np.random.default_rng(args.seed + label * 1009)
        # Use a separate deterministic stream for the official test split.
        test_rng = np.random.default_rng(args.seed + 500000 + label * 1009)
        train_rng.shuffle(train_candidates)
        test_rng.shuffle(test_candidates)

        if len(train_candidates) < args.wm_train_size:
            raise RuntimeError(
                f"MNIST official train label {label} has only {len(train_candidates)} samples; "
                f"{args.wm_train_size} are required."
            )
        if len(test_candidates) < args.wm_test_size:
            raise RuntimeError(
                f"MNIST official test label {label} has only {len(test_candidates)} samples; "
                f"{args.wm_test_size} are required."
            )

        train_indices = train_candidates[:args.wm_train_size].tolist()
        test_indices = test_candidates[:args.wm_test_size].tolist()

        train_set = Subset(train_mnist, train_indices)
        test_set = Subset(test_mnist, test_indices)

        wm_train_sets.append(train_set)
        wm_extra_sets.append(test_set)
        wm_test_sets.append(test_set)

        metadata["clients"][str(label)] = {
            "target_label": int(label),
            "train_source_split": "MNIST official train",
            "test_source_split": "MNIST official test",
            "train_mnist_indices": [int(x) for x in train_indices],
            "test_mnist_indices": [int(x) for x in test_indices],
            "train_test_overlap": 0,
            "split_disjoint_by_official_partition": True,
        }

    return wm_train_sets, wm_extra_sets, wm_test_sets, target_labels, metadata










def iid_partition(dataset: Dataset, num_clients: int, seed: int) -> Dict[int, List[int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return {i: splits[i].tolist() for i in range(num_clients)}


def dirichlet_partition(dataset: Dataset, num_clients: int, gamma: float, seed: int, min_size: int = 10):
    rng = np.random.default_rng(seed)
    targets = np.asarray(dataset.targets)
    num_classes = int(targets.max()) + 1
    while True:
        client_indices = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            idx_c = np.where(targets == c)[0]
            rng.shuffle(idx_c)
            proportions = rng.dirichlet(np.repeat(gamma, num_clients))
            split_points = (np.cumsum(proportions)[:-1] * len(idx_c)).astype(int)
            for i, part in enumerate(np.split(idx_c, split_points)):
                client_indices[i].extend(part.tolist())
        if min(len(x) for x in client_indices) >= min_size:
            break
    for i in range(num_clients):
        rng.shuffle(client_indices[i])
    return {i: client_indices[i] for i in range(num_clients)}


def trainable_params(model: nn.Module) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def is_surgery_param_name(name: str, args) -> bool:
    return name.startswith("fc2.") or name.startswith("classifier.6.")


def selected_surgery_params(model: nn.Module, args) -> List[nn.Parameter]:
    params = [p for name, p in model.named_parameters() if p.requires_grad and is_surgery_param_name(name, args)]
    if not params:
        raise RuntimeError("Final classification layer was not found; check the model definition.")
    return params


def build_surgery_indices(model: nn.Module, args) -> torch.Tensor:
    indices = []
    offset = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        if is_surgery_param_name(name, args):
            indices.append(torch.arange(offset, offset + n, dtype=torch.long))
        offset += n
    if not indices:
        raise RuntimeError("Final classification layer was not found; cannot build surgery indices.")
    return torch.cat(indices).to(args.device)


def flatten_grads(grads, params: List[nn.Parameter]) -> torch.Tensor:
    flat = []
    for g, p in zip(grads, params):
        if g is None:
            flat.append(torch.zeros_like(p, memory_format=torch.contiguous_format).view(-1))
        else:
            flat.append(g.contiguous().view(-1))
    return torch.cat(flat)


def replace_selected_grads(params: List[nn.Parameter], flat_grad: torch.Tensor):
    pointer = 0
    for p in params:
        n = p.numel()
        new_grad = flat_grad[pointer:pointer + n].view_as(p).detach().clone()
        if p.grad is None:
            p.grad = new_grad
        else:
            p.grad.detach().copy_(new_grad)
        pointer += n


def normalize_or_none(v: torch.Tensor, eps: float) -> Optional[torch.Tensor]:
    n = torch.norm(v, p=2)
    if n.item() <= eps or not torch.isfinite(n):
        return None
    return v / (n + eps)


def update_prototypes(prototypes: List[Optional[torch.Tensor]], avg_wm_grads: List[torch.Tensor], args):
    for i, g in enumerate(avg_wm_grads):
        if g.numel() == 0:
            continue
        g = g.to(args.device).float()
        if prototypes[i] is None:
            prototypes[i] = normalize_or_none(g, args.eps)
        else:
            mixed = args.ema_rho * prototypes[i] + (1.0 - args.ema_rho) * g
            prototypes[i] = normalize_or_none(mixed, args.eps)


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.dot(a, b) / (torch.norm(a, p=2) * torch.norm(b, p=2) + eps)


def gradient_conflict_metrics(grads: List[torch.Tensor], eps: float):
    valid = [g.float() for g in grads if g.numel() > 0 and torch.norm(g.float(), p=2).item() > eps]
    if len(valid) < 2:
        return 0.0, 0.0, 0.0
    wgc_sum, cosine_sum, neg_count, total = 0.0, 0.0, 0, 0
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            c = cosine(valid[i], valid[j], eps).item()
            wgc_sum += max(0.0, -c)
            cosine_sum += c
            neg_count += int(c < 0)
            total += 1
    return wgc_sum / total, neg_count / total, cosine_sum / total


def is_cagc_method(method: str) -> bool:
    return method == "fedcagc"


def prepare_prototype_bank(client_id: int, prototypes: List[Optional[torch.Tensor]], args):
    """Stack the fixed per-round prototype references once per client.

    This avoids rebuilding Python candidate lists and repeatedly synchronizing
    CUDA inside every local minibatch. The returned bank is fixed for the whole
    local update, matching the original round_prototypes semantics.
    """
    rows = []
    for j, h in enumerate(prototypes):
        if j == client_id or h is None:
            continue
        rows.append(h)
    if not rows:
        return None
    return torch.stack(rows, dim=0).to(device=args.device, dtype=torch.float32)


def apply_fedcagc_correction(client_id: int, g_part: torch.Tensor,
                             prototypes: List[Optional[torch.Tensor]], args,
                             prototype_bank: Optional[torch.Tensor] = None):
    """Apply finalized FedCAGC without host-side CUDA synchronization.

    Semantics are the same as the original implementation:
      1) initial negative-conflict candidates are determined from the
         uncorrected current watermark gradient;
      2) candidates are sorted from most negative cosine upward;
      3) after every projection, the conflict is re-checked;
      4) only initially-negative candidates can be processed;
      5) bounded norm compensation is applied at the end.

    The original code called .item() many times per minibatch, forcing GPU/CPU
    synchronization. Here the decisions stay on-device.
    """
    original = g_part
    g = g_part.clone()

    bank = prototype_bank
    if bank is None:
        bank = prepare_prototype_bank(client_id, prototypes, args)
    if bank is None or bank.numel() == 0:
        one = torch.ones((), device=g.device, dtype=g.dtype)
        return g, one, one

    # Initial cosine similarities determine the eligible candidate set.
    g_norm = torch.norm(g, p=2)
    h_norms = torch.norm(bank, p=2, dim=1)
    initial_cos = torch.mv(bank, g) / (h_norms * g_norm + args.eps)
    initial_negative = initial_cos < -args.tau_neg
    order = torch.argsort(initial_cos, dim=0, descending=False)
    eligible_sorted = initial_negative[order]

    # Preserve surgery_max_clients semantics without moving masks to CPU.
    if args.surgery_max_clients > 0:
        negative_rank = torch.cumsum(eligible_sorted.to(torch.int64), dim=0)
        eligible_sorted = eligible_sorted & (negative_rank <= args.surgery_max_clients)

    # Sequential re-check/project, but branch decisions remain CUDA tensors.
    for pos in range(bank.shape[0]):
        idx = order[pos]
        h = bank[idx]
        dot_gh = torch.dot(g, h)
        denom_cos = torch.norm(g, p=2) * torch.norm(h, p=2) + args.eps
        still_negative = (dot_gh / denom_cos) < -args.tau_neg
        do_project = eligible_sorted[pos] & still_negative
        projected = g - (dot_gh / (torch.dot(h, h) + args.eps)) * h
        g = torch.where(do_project, projected, g)

    n0 = torch.norm(original, p=2)
    n1 = torch.norm(g, p=2)
    retain_before_comp = n1 / (n0 + args.eps)

    if args.wm_compensate:
        valid = (n0 > args.eps) & (n1 > args.eps)
        scale = torch.clamp(n0 / (n1 + args.eps), max=args.max_comp_scale)
        scale = torch.where(valid, scale, torch.ones_like(scale))
        g = g * scale

    retain_after = torch.norm(g, p=2) / (n0 + args.eps)
    return g, retain_before_comp.detach(), retain_after.detach()



@torch.no_grad()
def measure_watermark_train_losses(model: nn.Module, wm_train_sets: List[Dataset], args) -> np.ndarray:
    """FedAWM reproduction signal; uses TRAIN watermark sets only (no test leakage)."""
    was_training = model.training
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    losses = []
    for wm_set in wm_train_sets:
        loader = DataLoader(wm_set, batch_size=256, shuffle=False, num_workers=0,
                            pin_memory=(args.device.type == "cuda"))
        total_loss, total = 0.0, 0
        for x, y in loader:
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            total_loss += criterion(model(x), y).item()
            total += y.size(0)
        losses.append(total_loss / max(total, 1))
    if was_training:
        model.train()
    return np.asarray(losses, dtype=np.float64)


def bounded_mean_one_scales(values: np.ndarray, args) -> np.ndarray:
    """Established project FedAWM reproduction: harder watermark sets receive larger bounded depth/scale."""
    values = np.asarray(values, dtype=np.float64)
    std = float(values.std())
    if std <= args.eps:
        return np.ones_like(values)
    standardized = (values - values.mean()) / (std + args.eps)
    logits = standardized / args.awm_temperature
    logits -= logits.max()
    raw = np.exp(logits)
    raw = raw / raw.sum() * len(raw)
    scales = np.clip(raw, args.awm_min_scale, args.awm_max_scale)
    for _ in range(8):
        scales = scales / max(scales.mean(), args.eps)
        scales = np.clip(scales, args.awm_min_scale, args.awm_max_scale)
    return scales.astype(np.float64)

def train_client(client_id: int, global_model: nn.Module, task_dataset: Dataset, wm_dataset: Dataset,
                 prototypes: List[Optional[torch.Tensor]],
                 use_watermark: bool, use_surgery: bool, round_id: int, args):
    local_model = copy.deepcopy(global_model).to(args.device)
    local_model.train()
    surgery_params = selected_surgery_params(local_model, args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(local_model.parameters(), lr=args.local_lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    task_generator = torch.Generator()
    task_generator.manual_seed(args.seed + round_id * 100000 + client_id * 1000 + 11)
    task_loader = DataLoader(
        task_dataset, batch_size=args.local_bs, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.device.type == "cuda"),
        drop_last=False, generator=task_generator
    )

    if not use_watermark:
        for _ in range(args.local_epochs):
            for x, y in task_loader:
                x = x.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(local_model(x), y)
                loss.backward()
                optimizer.step()
        z = torch.zeros(0, dtype=torch.float32)
        return copy.deepcopy(local_model.state_dict()), len(task_dataset), z, z, z, 0.0, 0.0

    # V2 uses all 100 watermark-training samples as one full watermark batch.
    # This preserves the original full-trigger-set watermark objective while avoiding
    # an imbalanced 64/36 minibatch cycle that would overweight the smaller batch.
    # The set is tiny, so caching it once on the GPU has negligible memory cost.
    wm_loader = DataLoader(
        wm_dataset,
        batch_size=len(wm_dataset),
        shuffle=False,
        num_workers=0,
        pin_memory=(args.device.type == "cuda"),
        drop_last=False,
    )
    cached_wm_batches = [
        (x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True))
        for x, y in wm_loader
    ]
    if not cached_wm_batches:
        raise RuntimeError("Watermark dataset is empty.")

    # Prototypes are fixed during a client's local update. Stack once rather
    # than reconstructing candidate lists in every minibatch.
    prototype_bank = prepare_prototype_bank(client_id, prototypes, args) if use_surgery else None

    raw_sum = post_sum = hist_sum = None
    retain_before_sum = torch.zeros((), device=args.device, dtype=torch.float32)
    retain_after_sum = torch.zeros((), device=args.device, dtype=torch.float32)
    count = 0

    for _ in range(args.local_epochs):
        wm_batch_id = 0
        for x_task, y_task in task_loader:
            x_task = x_task.to(args.device, non_blocking=True)
            y_task = y_task.to(args.device, non_blocking=True)

            x_wm, y_wm = cached_wm_batches[wm_batch_id]
            wm_batch_id = (wm_batch_id + 1) % len(cached_wm_batches)

            optimizer.zero_grad(set_to_none=True)
            loss_task = criterion(local_model(x_task), y_task)
            loss_wm = criterion(local_model(x_wm), y_wm)
            beta = args.wm_beta

            # One watermark-gradient query is still required by the method.
            grads_wm_part = torch.autograd.grad(
                loss_wm, surgery_params,
                retain_graph=True, create_graph=False, allow_unused=True
            )
            raw_part = flatten_grads(grads_wm_part, surgery_params)
            post_part = raw_part
            r_before = torch.ones((), device=args.device, dtype=raw_part.dtype)
            r_after = r_before

            if is_cagc_method(args.method) and use_surgery:
                fixed_part, r_before, r_after = apply_fedcagc_correction(
                    client_id, raw_part, prototypes, args,
                    prototype_bank=prototype_bank,
                )
                post_part = fixed_part

                # Original code performed a second autograd.grad(loss_task, ...)
                # and then a full backward. That duplicates backprop through the
                # task graph. Instead, do the required full backward once; its
                # selected-layer gradient is task_grad + beta * raw_wm_grad.
                # Replacing it by:
                #   combined_grad + beta * (fixed_wm_grad - raw_wm_grad)
                # is algebraically the same task_grad + beta * fixed_wm_grad.
                (loss_task + beta * loss_wm).backward()
                combined_part = flatten_grads(
                    [p.grad for p in surgery_params], surgery_params
                )
                corrected_combined = combined_part + beta * (fixed_part - raw_part)
                replace_selected_grads(surgery_params, corrected_combined)
            else:
                (loss_task + beta * loss_wm).backward()

            if raw_sum is None:
                raw_sum = torch.zeros_like(raw_part, dtype=torch.float32, device=args.device)
                post_sum = torch.zeros_like(post_part, dtype=torch.float32, device=args.device)
                hist_sum = torch.zeros_like(raw_part, dtype=torch.float32, device=args.device)
            raw_sum.add_(raw_part.float())
            post_sum.add_(post_part.float())
            hist_sum.add_(raw_part.float())
            retain_before_sum.add_(r_before.float())
            retain_after_sum.add_(r_after.float())
            count += 1

            if args.grad_clip_norm and args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), args.grad_clip_norm)
            optimizer.step()

    if count == 0:
        z = torch.zeros(0, dtype=torch.float32)
        return copy.deepcopy(local_model.state_dict()), len(task_dataset), z, z, z, 0.0, 0.0

    # Only two scalar synchronizations per client, rather than many per minibatch.
    retain_before_value = (retain_before_sum / count).item()
    retain_after_value = (retain_after_sum / count).item()
    return (
        copy.deepcopy(local_model.state_dict()),
        len(task_dataset),
        (hist_sum / count).detach().cpu(),
        (raw_sum / count).detach().cpu(),
        (post_sum / count).detach().cpu(),
        retain_before_value,
        retain_after_value,
    )



def train_client_flwb(client_id: int, global_model: nn.Module, task_dataset: Dataset,
                      wm_dataset: Dataset, round_id: int, args):
    """FLWB stepwise training: task update, then separated/enhanced watermark update."""
    local_model = copy.deepcopy(global_model).to(args.device)
    local_model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(local_model.parameters(), lr=args.local_lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    selected = selected_surgery_params(local_model, args)
    all_params = trainable_params(local_model)

    gen = torch.Generator(); gen.manual_seed(args.seed + round_id * 100000 + client_id * 1000 + 11)
    task_loader = DataLoader(task_dataset, batch_size=args.local_bs, shuffle=True,
                             num_workers=args.num_workers, pin_memory=(args.device.type == "cuda"),
                             drop_last=False, generator=gen)
    wm_loader = DataLoader(wm_dataset, batch_size=len(wm_dataset), shuffle=False,
                           num_workers=0, pin_memory=(args.device.type == "cuda"), drop_last=False)
    x_wm, y_wm = next(iter(wm_loader))
    x_wm = x_wm.to(args.device, non_blocking=True); y_wm = y_wm.to(args.device, non_blocking=True)

    raw_sum = post_sum = None
    count = 0
    for _ in range(args.local_epochs):
        for x_task, y_task in task_loader:
            x_task = x_task.to(args.device, non_blocking=True)
            y_task = y_task.to(args.device, non_blocking=True)
            # Step 1: ordinary task update w_r.
            optimizer.zero_grad(set_to_none=True)
            loss_task = criterion(local_model(x_task), y_task)
            loss_task.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), args.grad_clip_norm)
            optimizer.step()

            raw_steps = []
            for _wm_step in range(args.flwb_wm_steps):
                local_model.zero_grad(set_to_none=True)
                loss_wm = criterion(local_model(x_wm), y_wm)
                grads = torch.autograd.grad(loss_wm, all_params, retain_graph=False,
                                            create_graph=False, allow_unused=True)
                grad_map = {id(p): g for p, g in zip(all_params, grads)}
                selected_grads = [grad_map.get(id(p)) for p in selected]
                raw_part = flatten_grads(selected_grads, selected).detach()
                raw_steps.append(raw_part)

                # Algorithm-5 style delta enhancement. Keep the established reproduction setting's
                # established reproduction: one explicit watermark-gradient step per task batch.
                clip_coef = 1.0
                if args.grad_clip_norm > 0:
                    sq = [torch.sum(g.detach().float() ** 2) for g in grads if g is not None]
                    if sq:
                        total_norm = torch.sqrt(torch.stack(sq).sum())
                        clip_coef = torch.clamp(args.grad_clip_norm / (total_norm + args.eps), max=1.0).item()
                step_scale = args.wm_beta * args.flwb_lambda
                with torch.no_grad():
                    for p, g in zip(all_params, grads):
                        if g is not None:
                            p.add_(g, alpha=-args.local_lr * step_scale * clip_coef)

            raw_part = torch.stack(raw_steps).mean(dim=0)
            post_part = args.wm_beta * args.flwb_lambda * raw_part
            if raw_sum is None:
                raw_sum = torch.zeros_like(raw_part, dtype=torch.float32, device=args.device)
                post_sum = torch.zeros_like(post_part, dtype=torch.float32, device=args.device)
            raw_sum.add_(raw_part.float()); post_sum.add_(post_part.float()); count += 1

    if count == 0:
        z = torch.zeros(0, dtype=torch.float32)
        return copy.deepcopy(local_model.state_dict()), len(task_dataset), z, z, z, 1.0, 1.0
    avg_raw = (raw_sum / count).detach().cpu()
    avg_post = (post_sum / count).detach().cpu()
    return copy.deepcopy(local_model.state_dict()), len(task_dataset), avg_raw, avg_raw, avg_post, 1.0, 1.0


def train_client_by_method(client_id: int, global_model: nn.Module, task_dataset: Dataset,
                           wm_dataset: Dataset, prototypes: List[Optional[torch.Tensor]],
                           use_surgery: bool, round_id: int, allocation_scale: float, args):
    if args.method == "fedavg":
        return train_client(client_id, global_model, task_dataset, wm_dataset, prototypes,
                            False, False, round_id, args)
    if args.method == "flwb":
        return train_client_flwb(client_id, global_model, task_dataset, wm_dataset, round_id, args)

    local_args = copy.copy(args)
    if args.method == "fedawm":
        local_args.wm_beta = args.wm_beta * float(allocation_scale)
    # fedipr and fedcagc retain wm_beta=1.0 in the formal V2 protocol.
    return train_client(client_id, global_model, task_dataset, wm_dataset, prototypes,
                        True, bool(use_surgery), round_id, local_args)





def fedavg(client_states: List[Dict[str, torch.Tensor]], client_sizes: List[int]):
    total_size = float(sum(client_sizes))
    avg_state = copy.deepcopy(client_states[0])
    for key in avg_state.keys():
        if torch.is_floating_point(avg_state[key]):
            avg_state[key] = torch.zeros_like(avg_state[key])
            for state, size in zip(client_states, client_sizes):
                avg_state[key] += state[key] * (size / total_size)
        else:
            avg_state[key] = client_states[0][key]
    return avg_state


@torch.no_grad()
def evaluate_task(model: nn.Module, dataset: Dataset, args) -> Tuple[float, float]:
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(args.device.type == "cuda"))
    criterion = nn.CrossEntropyLoss(reduction="sum")
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x = x.to(args.device, non_blocking=True)
        y = y.to(args.device, non_blocking=True)
        logits = model(x)
        loss_sum += criterion(logits, y).item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total, loss_sum / total


@torch.no_grad()
def evaluate_watermarks(model: nn.Module, wm_test_sets: List[Dataset], target_labels: List[int], args):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    per_client_wma = np.zeros(len(wm_test_sets), dtype=np.float64)
    wm_loss_sum, wm_total, cross_sum, cross_total = 0.0, 0, 0.0, 0
    for i, wm_set in enumerate(wm_test_sets):
        loader = DataLoader(wm_set, batch_size=256, shuffle=False,
                            num_workers=0, pin_memory=(args.device.type == "cuda"))
        client_correct, client_total = 0, 0
        for x, y in loader:
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=1)
            wm_loss_sum += criterion(logits, y).item()
            wm_total += y.size(0)
            client_correct += (pred == y).sum().item()
            client_total += y.size(0)
            for j, yj in enumerate(target_labels):
                if j != i:
                    cross_sum += (pred == yj).sum().item()
                    cross_total += y.size(0)
        per_client_wma[i] = client_correct / client_total
    wma = float(np.mean(per_client_wma))
    min_wma = float(np.min(per_client_wma))
    wm_loss = wm_loss_sum / wm_total
    cross_wma = cross_sum / cross_total if cross_total > 0 else 0.0
    return per_client_wma, wma, min_wma, wm_loss, cross_wma


@torch.no_grad()
def evaluate_verification_matrix(model: nn.Module, wm_sets: List[Dataset],
                                 target_labels: List[int], args) -> np.ndarray:
    """Rows are trigger owners; columns are claimed client identities."""
    model.eval()
    matrix = np.zeros((len(wm_sets), len(target_labels)), dtype=np.float64)
    for i, wm_set in enumerate(wm_sets):
        loader = DataLoader(
            wm_set, batch_size=256, shuffle=False, num_workers=0,
            pin_memory=(args.device.type == "cuda")
        )
        counts = np.zeros(len(target_labels), dtype=np.float64)
        total = 0
        for x, _ in loader:
            x = x.to(args.device, non_blocking=True)
            pred = model(x).argmax(dim=1)
            total += pred.numel()
            for j, target_label in enumerate(target_labels):
                counts[j] += (pred == target_label).sum().item()
        if total > 0:
            matrix[i] = counts / total
    return matrix


def evaluate_global_gradient_conflicts(model: nn.Module, wm_train_sets: List[Dataset],
                                       prototypes: List[Optional[torch.Tensor]],
                                       use_correction: bool, args):
    """Evaluate WGC on one aggregated model and one fixed full watermark batch per client.

    The model is put in evaluation mode so dropout and data augmentation cannot make the
    pre/post comparison use different stochastic gradients. No optimizer state is changed.
    """
    was_training = model.training
    model.eval()
    params = selected_surgery_params(model, args)
    criterion = nn.CrossEntropyLoss()
    raw_grads = []
    for wm_set in wm_train_sets:
        loader = DataLoader(
            wm_set, batch_size=len(wm_set), shuffle=False, num_workers=0,
            pin_memory=(args.device.type == "cuda"), drop_last=False
        )
        x, y = next(iter(loader))
        x = x.to(args.device, non_blocking=True)
        y = y.to(args.device, non_blocking=True)
        loss = criterion(model(x), y)
        grads = torch.autograd.grad(
            loss, params, retain_graph=False, create_graph=False, allow_unused=True
        )
        raw_grads.append(flatten_grads(grads, params).detach())

    if use_correction:
        corrected_grads = [
            apply_fedcagc_correction(client_id, grad, prototypes, args)[0].detach()
            for client_id, grad in enumerate(raw_grads)
        ]
    else:
        corrected_grads = [grad.clone() for grad in raw_grads]

    pre_wgc, pre_rate, pre_mean_cos = gradient_conflict_metrics(raw_grads, args.eps)
    post_wgc, post_rate, post_mean_cos = gradient_conflict_metrics(corrected_grads, args.eps)
    if was_training:
        model.train()
    return {
        "raw_grads": [grad.cpu() for grad in raw_grads],
        "corrected_grads": [grad.cpu() for grad in corrected_grads],
        "wgc_pre": pre_wgc,
        "wgc_post": post_wgc,
        "conflict_rate_pre": pre_rate,
        "conflict_rate_post": post_rate,
        "mean_cos_pre": pre_mean_cos,
        "mean_cos_post": post_mean_cos,
    }


def json_ready(value):
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_ready(value), file, ensure_ascii=False, indent=2)


def cpu_prototypes(prototypes: List[Optional[torch.Tensor]]):
    return [None if prototype is None else prototype.detach().cpu() for prototype in prototypes]


def save_matrix_csv(path: Path, matrix: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["trigger_owner"] + [f"claimed_client_{j}" for j in range(matrix.shape[1])])
        for i, row in enumerate(matrix):
            writer.writerow([f"client_{i}"] + row.tolist())


def save_history_csv(history: List[Dict], args):
    csv_dir = os.path.dirname(args.csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    fieldnames = [
        "dataset", "method", "round", "mta", "main_loss",
        "wma", "min_wma", "std_wma", "max_wma", "gap_wma", "wm_loss", "cross_wma",
        "wma_train", "min_wma_train", "wma_extra", "min_wma_extra",
        "eval_wgc_pre", "eval_wgc_post", "eval_conflict_rate_pre", "eval_conflict_rate_post",
        "eval_mean_cos_pre", "eval_mean_cos_post",
        "train_wgc_pre", "train_wgc_post", "train_conflict_rate_pre", "train_conflict_rate_post",
        "train_mean_cos_pre", "train_mean_cos_post",
        "wm_retain_before_comp", "wm_retain_after_comp", "round_seconds",
        "allocation_mean", "allocation_min", "allocation_max",
        "num_clients", "non_iid", "gamma", "surgery_scope", "ema_rho", "tau_neg",
        "wm_beta", "wm_train_size", "wm_test_size", "wm_compensate", "max_comp_scale",
        "grad_clip_norm", "watermark_source", "num_outputs", "flwb_lambda", "flwb_wm_steps",
        "awm_temperature", "awm_min_scale", "awm_max_scale", "awm_ema",
        "seed",
    ]
    fieldnames.extend([f"wma_client_{i}" for i in range(args.num_clients)])
    fieldnames.extend([f"allocation_client_{i}" for i in range(args.num_clients)])
    with open(args.csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            allocation = np.asarray(record.get("allocation", np.ones(args.num_clients)), dtype=np.float64)
            row = {k: record.get(k, getattr(args, k, "")) for k in fieldnames
                   if not k.startswith("wma_client_") and not k.startswith("allocation_client_")}
            row.update({
                "dataset": args.dataset, "method": args.method, "num_clients": args.num_clients,
                "non_iid": int(args.non_iid), "gamma": args.gamma, "surgery_scope": args.surgery_scope,
                "ema_rho": args.ema_rho, "tau_neg": args.tau_neg, "wm_beta": args.wm_beta,
                "wm_train_size": args.wm_train_size, "wm_test_size": args.wm_test_size,
                "wm_compensate": int(args.wm_compensate), "max_comp_scale": args.max_comp_scale,
                "grad_clip_norm": args.grad_clip_norm, "watermark_source": args.watermark_source,
                "num_outputs": args.num_outputs, "flwb_lambda": args.flwb_lambda,
                "flwb_wm_steps": args.flwb_wm_steps, "awm_temperature": args.awm_temperature,
                "awm_min_scale": args.awm_min_scale, "awm_max_scale": args.awm_max_scale,
                "awm_ema": args.awm_ema, "seed": args.seed,
                "allocation_mean": float(allocation.mean()), "allocation_min": float(allocation.min()),
                "allocation_max": float(allocation.max()),
            })
            for i, value in enumerate(record["per_client_wma"]):
                row[f"wma_client_{i}"] = value
            for i, value in enumerate(allocation):
                row[f"allocation_client_{i}"] = value
            writer.writerow(row)
    print(f"CSV saved to              : {args.csv_path}")


def main():
    args = parse_args()
    fix_random_seed(args.seed)
    if args.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.fast_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.fast_tf32)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    references = {
        "fedipr": "Li et al., FedIPR: Ownership Verification for Federated Deep Neural Network Models, IEEE TPAMI, DOI 10.1109/TPAMI.2022.3195956",
        "flwb": "Li et al., Federated Learning Watermark Based on Model Backdoor, Journal of Software 2024, DOI 10.13328/j.cnki.jos.006914",
        "fedawm": "Sun et al., FedAWM: Adaptive Watermark Allocation in Non-IID Federated Learning, Knowledge-Based Systems 332:114938, DOI 10.1016/j.knosys.2025.114938",
        "fedcagc": "FedCAGC",
    }
    config = dict(vars(args))
    config.update({
        "script": "fedcagc_v2_all_methods.py",
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "references": references,
        "comparison_protocol": {
            "fedipr_flwb_fedawm_fedcagc_watermark": "same official MNIST train100/test200 trigger sets and target labels, per unified comparison protocol fairness protocol",
            "fedipr_scope": "black-box/backdoor branch only; white-box feature watermark excluded",
            "fedawm_scope": "established project reproduction carried forward unchanged in mechanism; allocation uses watermark-training loss only and never test data",
        },
        "fedcagc_constraints": {
            "negative_conflict_only": True,
            "positive_overlap_reduction": False,
            "first_round_cold_start": True,
            "prototype_source": "uncorrected_raw_watermark_gradients",
            "prototype_update": "EMA then L2 normalization",
            "server_aggregation": "weighted FedAvg unchanged",
        },
    })
    save_json(output_dir / "config.json", config)

    print("========== FedCAGC V2 Unified Comparison ==========")
    for name in [
        "dataset", "method", "device", "num_clients", "num_outputs", "rounds", "local_epochs",
        "local_bs", "local_lr", "non_iid", "gamma", "wm_beta", "wm_train_size", "wm_test_size",
        "grad_clip_norm", "seed",
    ]:
        print(f"{name:24s}: {getattr(args, name)}")
    if args.method == "flwb":
        print(f"{'flwb_lambda':24s}: {args.flwb_lambda}")
    if args.method == "fedawm":
        print(f"{'awm scale range':24s}: [{args.awm_min_scale}, {args.awm_max_scale}]")
    print(f"{'output_dir':24s}: {output_dir}")

    train_dataset, test_dataset = get_main_datasets(args)
    wm_train_sets, wm_extra_sets, wm_test_sets, target_labels, wm_metadata = build_watermark_datasets(args)

    user_groups = dirichlet_partition(train_dataset, args.num_clients, args.gamma, args.seed) if args.non_iid else iid_partition(train_dataset, args.num_clients, args.seed)
    if args.smoke_test:
        print("\nWARNING: smoke_test is enabled; these results are only for code-path validation.")
        user_groups = {cid: idxs[:min(len(idxs), args.smoke_samples_per_client)] for cid, idxs in user_groups.items()}
        test_dataset = Subset(test_dataset, list(range(min(len(test_dataset), args.smoke_test_samples))))
    client_task_sets = [Subset(train_dataset, user_groups[i]) for i in range(args.num_clients)]
    save_json(output_dir / "client_partition.json", user_groups)
    save_json(output_dir / "watermark_split.json", wm_metadata)

    print("\nClient task data sizes:")
    for i, ds in enumerate(client_task_sets):
        print(f"  Client {i}: {len(ds)}")
    print("\nWatermark datasets:")
    for i in range(args.num_clients):
        source_desc = f"MNIST label={i}" if args.watermark_source == "mnist" else f"Waffle id={i}"
        print(f"  Client {i}: {source_desc}, target={target_labels[i]}, train={len(wm_train_sets[i])}, test={len(wm_test_sets[i])}")

    global_model = get_model(args.dataset, args.num_outputs).to(args.device)
    total_params = sum(p.numel() for p in trainable_params(global_model))
    surgery_indices = build_surgery_indices(global_model, args)
    model_bytes = sum(p.numel() * p.element_size() for p in trainable_params(global_model))
    q = int(surgery_indices.numel())
    elem_bytes = next(global_model.parameters()).element_size()
    server_prototype_bytes = args.num_clients * q * elem_bytes if is_cagc_method(args.method) else 0
    extra_upload_bytes = args.rounds * args.num_clients * q * elem_bytes if is_cagc_method(args.method) else 0
    extra_download_bytes = max(0, args.rounds - 1) * args.num_clients * args.num_clients * q * elem_bytes if is_cagc_method(args.method) else 0
    print(f"\nTrainable parameter dimension d = {total_params}")
    print(f"Final-classifier gradient dimension q = {q} ({q / total_params:.4%})")

    prototypes: List[Optional[torch.Tensor]] = [None for _ in range(args.num_clients)]
    awm_scales = np.ones(args.num_clients, dtype=np.float64)
    history, prototype_snapshots, gradient_snapshots = [], {}, {}
    total_start = time.perf_counter()
    if args.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)

    for rnd in range(1, args.rounds + 1):
        if args.device.type == "cuda": torch.cuda.synchronize(args.device)
        round_start = time.perf_counter()
        print(f"\n-------- Round {rnd:03d}/{args.rounds:03d} --------")

        use_surgery = is_cagc_method(args.method) and rnd >= 2 and all(p is not None for p in prototypes)
        round_prototypes = [None if p is None else p.detach().clone() for p in prototypes]
        if args.method == "fedcagc" and rnd == 1:
            print("Cold-start round: no correction; collect raw watermark gradients for historical prototypes.")
        elif args.method == "fedcagc" and rnd == 2:
            print(f"FedCAGC correction starts: {use_surgery}")

        # FedAWM reproduction allocation is derived only from watermark TRAIN sets.
        if args.method == "fedawm":
            losses = measure_watermark_train_losses(global_model, wm_train_sets, args)
            proposed = bounded_mean_one_scales(losses, args)
            awm_scales = args.awm_ema * awm_scales + (1.0 - args.awm_ema) * proposed
            awm_scales = np.clip(awm_scales / max(awm_scales.mean(), args.eps), args.awm_min_scale, args.awm_max_scale)
            awm_scales = awm_scales / max(awm_scales.mean(), args.eps)
            allocation = awm_scales.copy()
            print(f"FedAWM allocation mean/min/max: {allocation.mean():.4f}/{allocation.min():.4f}/{allocation.max():.4f}")
        else:
            allocation = np.ones(args.num_clients, dtype=np.float64)

        client_states, client_sizes = [], []
        hist_grads, raw_grads, post_grads = [], [], []
        retain_before, retain_after = [], []
        train_phase_start = time.perf_counter()
        for cid in range(args.num_clients):
            state, size, hist_g, raw_g, post_g, r_before, r_after = train_client_by_method(
                cid, global_model, client_task_sets[cid], wm_train_sets[cid], round_prototypes,
                use_surgery, rnd, float(allocation[cid]), args
            )
            client_states.append(state); client_sizes.append(size)
            if args.method != "fedavg":
                hist_grads.append(hist_g); raw_grads.append(raw_g); post_grads.append(post_g)
                retain_before.append(r_before); retain_after.append(r_after)
        if args.device.type == "cuda": torch.cuda.synchronize(args.device)
        train_phase_seconds = time.perf_counter() - train_phase_start

        aggregate_phase_start = time.perf_counter()
        global_model.load_state_dict(fedavg(client_states, client_sizes))
        if args.method == "fedcagc":
            update_prototypes(prototypes, hist_grads, args)
            if rnd == 1:
                print(f"Round-1 prototypes initialized: {sum(p is not None for p in prototypes)}/{args.num_clients}")
        if rnd in args.snapshot_rounds:
            prototype_snapshots[rnd] = cpu_prototypes(prototypes)

        if args.device.type == "cuda": torch.cuda.synchronize(args.device)
        aggregate_phase_seconds = time.perf_counter() - aggregate_phase_start

        train_wgc_pre, train_rate_pre, train_mean_cos_pre = gradient_conflict_metrics(raw_grads, args.eps)
        train_wgc_post, train_rate_post, train_mean_cos_post = gradient_conflict_metrics(post_grads, args.eps)
        mean_retain_before = float(np.mean(retain_before)) if retain_before else 0.0
        mean_retain_after = float(np.mean(retain_after)) if retain_after else 0.0

        tail_start = max(1, args.rounds - args.eval_tail + 1) if args.eval_tail > 0 else args.rounds + 1
        should_eval = rnd == 1 or rnd % args.eval_every == 0 or rnd >= tail_start or rnd in args.snapshot_rounds or rnd == args.rounds
        if should_eval:
            eval_phase_start = time.perf_counter()
            mta, main_loss = evaluate_task(global_model, test_dataset, args)
            if args.method == "fedavg":
                per_client_wma = np.full(args.num_clients, np.nan, dtype=np.float64)
                wma = min_wma = wm_loss = cross_wma = float("nan")
                wma_train = min_wma_train = float("nan")
                wma_extra = min_wma_extra = float("nan")
                global_conflicts = {
                    "wgc_pre": float("nan"), "wgc_post": float("nan"),
                    "conflict_rate_pre": float("nan"), "conflict_rate_post": float("nan"),
                    "mean_cos_pre": float("nan"), "mean_cos_post": float("nan"),
                    "raw_grads": [], "corrected_grads": [],
                }
            else:
                per_client_wma, wma, min_wma, wm_loss, cross_wma = evaluate_watermarks(
                    global_model, wm_test_sets, target_labels, args
                )
                if rnd == args.rounds:
                    _, wma_train, min_wma_train, _, _ = evaluate_watermarks(
                        global_model, wm_train_sets, target_labels, args
                    )
                else:
                    wma_train = min_wma_train = float("nan")
                wma_extra, min_wma_extra = wma, min_wma
                global_conflicts = evaluate_global_gradient_conflicts(
                    global_model, wm_train_sets, round_prototypes,
                    use_correction=use_surgery, args=args
                )
            if rnd in args.snapshot_rounds:
                gradient_snapshots[rnd] = {"raw": global_conflicts["raw_grads"], "corrected": global_conflicts["corrected_grads"]}
            if args.device.type == "cuda": torch.cuda.synchronize(args.device)
            round_seconds = time.perf_counter() - round_start
            eval_phase_seconds = time.perf_counter() - eval_phase_start
            std_wma = float(np.std(per_client_wma, ddof=0)); max_wma = float(np.max(per_client_wma))
            reported_cross = cross_wma
            record = {
                "round": rnd, "mta": mta, "main_loss": main_loss, "wma": wma, "min_wma": min_wma,
                "std_wma": std_wma, "max_wma": max_wma, "gap_wma": max_wma - min_wma,
                "wm_loss": wm_loss, "cross_wma": reported_cross,
                "wma_train": wma_train, "min_wma_train": min_wma_train,
                "wma_extra": wma_extra, "min_wma_extra": min_wma_extra,
                "eval_wgc_pre": global_conflicts["wgc_pre"], "eval_wgc_post": global_conflicts["wgc_post"],
                "eval_conflict_rate_pre": global_conflicts["conflict_rate_pre"], "eval_conflict_rate_post": global_conflicts["conflict_rate_post"],
                "eval_mean_cos_pre": global_conflicts["mean_cos_pre"], "eval_mean_cos_post": global_conflicts["mean_cos_post"],
                "train_wgc_pre": train_wgc_pre, "train_wgc_post": train_wgc_post,
                "train_conflict_rate_pre": train_rate_pre, "train_conflict_rate_post": train_rate_post,
                "train_mean_cos_pre": train_mean_cos_pre, "train_mean_cos_post": train_mean_cos_post,
                "wm_retain_before_comp": mean_retain_before, "wm_retain_after_comp": mean_retain_after,
                "round_seconds": round_seconds, "per_client_wma": per_client_wma.copy(), "allocation": allocation.copy(),
            }
            history.append(record); save_history_csv(history, args)
            print(f"MTA                  : {mta:.4f}")
            print(f"WMA test/train       : {wma:.4f} / {wma_train:.4f}")
            print(f"Min/Std/Gap WMA      : {min_wma:.4f} / {std_wma:.4f} / {max_wma-min_wma:.4f}")
            print(f"Eval WGC pre/post    : {global_conflicts['wgc_pre']:.4f} / {global_conflicts['wgc_post']:.4f}")
            print(f"Round time (s)       : {round_seconds:.2f}")
            if args.profile_round:
                print(f"Timing train/agg/eval: {train_phase_seconds:.2f} / {aggregate_phase_seconds:.2f} / {eval_phase_seconds:.2f} s")

    final = history[-1]
    print("\n========== Final Result ==========")
    print(f"Dataset               : {args.dataset}")
    print(f"Method                : {args.method}")
    print(f"Final MTA             : {final['mta']:.4f}")
    if args.method == "fedavg":
        print("Final WMA             : N/A")
        print("Final WGC pre/post    : N/A / N/A")
    else:
        print(f"Final WMA             : {final['wma']:.4f}")
        print(f"Final Min-WMA         : {final['min_wma']:.4f}")
        print(f"Final WGC pre/post    : {final['eval_wgc_pre']:.4f} / {final['eval_wgc_post']:.4f}")
        print("Final WMA per client  :", " ".join(f"{x:.4f}" for x in final["per_client_wma"]))
    save_history_csv(history, args)

    if args.method != "fedavg":
        verification_matrix = evaluate_verification_matrix(global_model, wm_test_sets, target_labels, args)
        np.save(output_dir / "verification_matrix.npy", verification_matrix)
        save_matrix_csv(output_dir / "verification_matrix.csv", verification_matrix)
    checkpoint_state = {key: value.detach().cpu() for key, value in global_model.state_dict().items()}
    torch.save({"model_state_dict": checkpoint_state, "prototypes": cpu_prototypes(prototypes),
                "config": json_ready(config), "final_record": json_ready(final)}, output_dir / "final_checkpoint.pt")
    torch.save(prototype_snapshots, output_dir / "prototype_snapshots.pt")
    torch.save(gradient_snapshots, output_dir / "gradient_snapshots.pt")

    if args.device.type == "cuda":
        torch.cuda.synchronize(args.device); peak_memory_bytes = int(torch.cuda.max_memory_allocated(args.device))
    else:
        peak_memory_bytes = 0
    total_seconds = time.perf_counter() - total_start
    diag = float(np.mean(np.diag(verification_matrix))) if verification_matrix.size else float("nan")
    offdiag = float((verification_matrix.sum() - np.trace(verification_matrix)) / max(1, verification_matrix.size - len(verification_matrix)))
    run_summary = {
        "dataset": args.dataset, "method": args.method, "seed": args.seed, "rounds": args.rounds,
        "final_mta": final["mta"], "final_wma": final["wma"], "final_min_wma": final["min_wma"],
        "final_std_wma": final["std_wma"], "final_wgc_pre": final["eval_wgc_pre"], "final_wgc_post": final["eval_wgc_post"],
        "total_seconds": total_seconds, "mean_recorded_round_seconds": float(np.mean([x["round_seconds"] for x in history])),
        "peak_gpu_memory_bytes": peak_memory_bytes, "model_parameter_count": total_params,
        "model_parameter_bytes": model_bytes, "final_layer_dimension_q": q,
        "server_prototype_storage_bytes": server_prototype_bytes,
        "estimated_extra_prototype_upload_bytes": extra_upload_bytes,
        "estimated_extra_prototype_download_bytes": extra_download_bytes,
        "verification_matrix_diagonal_mean": diag,
        "verification_matrix_off_diagonal_mean": offdiag,
    }
    save_json(output_dir / "run_summary.json", run_summary)
    print(f"Artifacts              : {output_dir}")


if __name__ == "__main__":
    main()
