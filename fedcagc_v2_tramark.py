#!/usr/bin/env python3
"""TraMark baseline under the FedCAGC V2 fair-comparison protocol.

Algorithmic source:
  Xu et al., "Traceable Black-Box Watermarks for Federated Learning", ICLR 2026.
  Official implementation: https://github.com/JiiahaoXU/TraMark

This runner preserves TraMark's method-specific mechanism while aligning the
public comparison protocol with FedCAGC V2:
  * same V2 model architectures and main-task transforms;
  * same saved Dirichlet client partition for each dataset/seed;
  * same MNIST sample identities used by V2: 100 official-train watermarks and
    200 disjoint official-test watermarks per client;
  * 10 clients, Local Epoch=5, batch=64, LR=0.01, momentum=0.9,
    weight decay=1e-4, Dirichlet gamma=0.5, 100 communication rounds.

TraMark-specific settings follow the ICLR 2026 paper / official code:
  * warmup ratio alpha=0.5;
  * watermark partition ratio k=0.01;
  * watermark injection LR=1e-4, epochs=5, batch=32, momentum=0;
  * personalized watermark region + masked aggregation;
  * MNIST label i is the watermark identity for client i.

Metric note:
TraMark creates a personalized model for each client. The formal comparison reports:
  MTA = mean main-task test accuracy across the personalized models;
  WMA = mean diagonal accuracy: model i on client i's independent watermark test set.
WGC is not reported because TraMark does not optimize multiple client watermarks in one shared parameter region.
"""

import argparse
import copy
import csv
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def parse_args():
    p = argparse.ArgumentParser(description="TraMark ICLR 2026 baseline under FedCAGC V2 protocol")
    p.add_argument("--project", default="/home/test/lxl/FedCAGC_V2")
    p.add_argument("--data_path", default="/home/test/lxl/data")
    p.add_argument("--dataset", choices=["fmnist", "cifar10"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=2)

    # V2 common protocol
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--local_epochs", type=int, default=5)
    p.add_argument("--local_bs", type=int, default=64)
    p.add_argument("--local_lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--wm_train_size", type=int, default=100)
    p.add_argument("--wm_test_size", type=int, default=200)

    # TraMark defaults from the paper / official code
    p.add_argument("--alpha", type=float, default=0.5, help="warmup training ratio")
    p.add_argument("--k", type=float, default=0.01, help="watermark parameter-region ratio")
    p.add_argument("--wm_epochs", type=int, default=5)
    p.add_argument("--wm_lr", type=float, default=1e-4)
    p.add_argument("--wm_momentum", type=float, default=0.0)
    p.add_argument("--wm_bs", type=int, default=32)
    p.add_argument(
        "--wm_grad_mode", choices=["official_accumulate", "zero_each_batch"],
        default="official_accumulate",
        help="Official repository does not zero watermark gradients between minibatches. "
             "Use official_accumulate for the formal reproduction."
    )
    p.add_argument(
        "--wm_transform", choices=["official", "v2"], default="official",
        help="Method-specific TraMark MNIST transform or the V2 common normalization. "
             "Formal TraMark reproduction should use official."
    )

    # Evaluation/output
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--eval_tail", type=int, default=10)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--save_checkpoint", action="store_true", default=False)
    return p.parse_args()


def load_v2_module(project: Path):
    src = project / "fedcagc_v2_all_methods.py"
    if not src.exists():
        raise FileNotFoundError(f"Missing V2 source: {src}")
    spec = importlib.util.spec_from_file_location("fedcagc_v2_all_methods_runtime", str(src))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def cpu_state_dict(model: nn.Module):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state(model: nn.Module, state: Dict[str, torch.Tensor], device):
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model


def save_json(path: Path, obj):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_rows(path: Path, rows: List[dict]):
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def should_eval(rnd: int, rounds: int, every: int, tail: int):
    return rnd == 1 or rnd == rounds or rnd % every == 0 or rnd > rounds - tail




def validate_against_v2_source(project: Path, args):
    cfg_path = project / "results" / "main" / f"{args.dataset}_fedcagc_{args.seed}" / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing V2 reference config: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    checks = {
        "num_clients": args.num_clients,
        "local_epochs": args.local_epochs,
        "local_bs": args.local_bs,
        "local_lr": args.local_lr,
        "gamma": args.gamma,
        "wm_train_size": args.wm_train_size,
        "wm_test_size": args.wm_test_size,
    }
    for key, requested in checks.items():
        recorded = cfg.get(key)
        if recorded is None:
            continue
        if isinstance(requested, float):
            ok = abs(float(recorded) - float(requested)) <= 1e-12
        else:
            ok = recorded == requested
        if not ok:
            raise RuntimeError(
                f"V2 protocol mismatch for {key}: requested={requested}, reference={recorded}. "
                "Use the same V2 common setting for the formal TraMark comparison."
            )
    if not bool(cfg.get("non_iid", True)):
        raise RuntimeError("Reference V2 run is not Non-IID; expected Dirichlet V2 protocol.")
    return cfg, cfg_path

def exact_v2_partition(project: Path, dataset: str, seed: int):
    p = project / "results" / "main" / f"{dataset}_fedcagc_{seed}" / "client_partition.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing V2 client partition: {p}. This baseline intentionally reuses the exact V2 partition."
        )
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return {int(k): [int(x) for x in v] for k, v in obj.items()}, p


def exact_v2_watermark_indices(project: Path, dataset: str, seed: int, num_clients: int,
                               train_n: int, test_n: int):
    p = project / "results" / "main" / f"{dataset}_fedcagc_{seed}" / "watermark_split.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing V2 watermark split: {p}. This baseline intentionally reuses the exact V2 sample identities."
        )
    obj = json.load(open(p, "r", encoding="utf-8"))
    out = {}
    for cid in range(num_clients):
        m = obj["clients"][str(cid)]
        tr = [int(x) for x in m["train_mnist_indices"]]
        te = [int(x) for x in m["test_mnist_indices"]]
        if len(tr) != train_n or len(te) != test_n:
            raise RuntimeError(
                f"V2 watermark size mismatch for client {cid}: train={len(tr)}, test={len(te)}; "
                f"expected {train_n}/{test_n}."
            )
        out[cid] = (tr, te)
    return out, p


def tramark_mnist_transform(dataset: str, mode: str, v2mod, v2args):
    if mode == "v2":
        return v2mod.get_mnist_watermark_transform(v2args)
    # Exact method-specific preprocessing used by the official TraMark repository.
    if dataset == "fmnist":
        return transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    if dataset == "cifar10":
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])
    raise ValueError(dataset)


def build_watermarks(args, v2mod, v2args, project: Path):
    split, split_path = exact_v2_watermark_indices(
        project, args.dataset, args.seed, args.num_clients, args.wm_train_size, args.wm_test_size
    )
    transform = tramark_mnist_transform(args.dataset, args.wm_transform, v2mod, v2args)
    tr_mnist = datasets.MNIST(args.data_path, train=True, download=True, transform=transform)
    te_mnist = datasets.MNIST(args.data_path, train=False, download=True, transform=transform)
    train_sets, test_sets = {}, {}
    meta = {
        "method": "TraMark",
        "source": "MNIST",
        "train_source": "MNIST official train",
        "test_source": "MNIST official test",
        "train_test_disjoint": True,
        "wm_transform": args.wm_transform,
        "source_split_file": str(split_path),
        "clients": {},
    }
    for cid in range(args.num_clients):
        tri, tei = split[cid]
        train_sets[cid] = Subset(tr_mnist, tri)
        test_sets[cid] = Subset(te_mnist, tei)
        meta["clients"][str(cid)] = {
            "target_label": cid,
            "train_indices": tri,
            "test_indices": tei,
        }
    return train_sets, test_sets, meta


def train_local_main(model: nn.Module, dataset, cid: int, rnd: int, args, device):
    model.train()
    opt = torch.optim.SGD(
        model.parameters(), lr=args.local_lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    crit = nn.CrossEntropyLoss()
    gen = torch.Generator()
    gen.manual_seed(args.seed + rnd * 100000 + cid * 1000 + 11)
    loader = DataLoader(
        dataset, batch_size=args.local_bs, shuffle=True, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), drop_last=False, generator=gen,
    )
    for _ in range(args.local_epochs):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
    return cpu_state_dict(model)


def fedavg(states: Dict[int, Dict[str, torch.Tensor]], sizes: Dict[int, int]):
    total = float(sum(sizes.values()))
    keys = list(next(iter(states.values())).keys())
    out = {}
    for k in keys:
        first = states[0][k]
        if torch.is_floating_point(first):
            acc = torch.zeros_like(first, dtype=torch.float32)
            for cid, st in states.items():
                acc.add_(st[k].float(), alpha=sizes[cid] / total)
            out[k] = acc.to(dtype=first.dtype)
        else:
            out[k] = first.clone()
    return out


def generate_masks(model: nn.Module, k: float):
    """Official TraMark per-tensor magnitude mask: top (1-k) -> main task; remainder -> WM."""
    if not (0.0 < k < 1.0):
        raise ValueError("TraMark k must be in (0,1).")
    main, wm = {}, {}
    for name, p in model.named_parameters():
        flat = p.detach().abs().view(-1).cpu()
        n = flat.numel()
        topn = int(n * (1.0 - k))
        mask = torch.zeros(n, dtype=torch.bool)
        if topn > 0:
            _, idx = torch.topk(flat, topn, largest=True)
            mask[idx] = True
        main[name] = mask.view_as(p.detach().cpu())
        wm[name] = ~main[name]
    return main, wm


def masked_aggregate(local_states, avg_state, main_mask, wm_mask, model: nn.Module):
    param_names = set(dict(model.named_parameters()).keys())
    personalized = {}
    for cid, st in local_states.items():
        ns = {}
        for name, avg_v in avg_state.items():
            if name in param_names:
                mm = main_mask[name]
                wm = wm_mask[name]
                ns[name] = torch.where(mm, avg_v, st[name]).clone()
            else:
                # Non-parameter state (if any) follows the aggregated main-task state.
                ns[name] = avg_v.clone()
        personalized[cid] = ns
    return personalized


def inject_watermark(model: nn.Module, wm_dataset, wm_mask, args, device, cid: int, rnd: int):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=args.wm_lr, momentum=args.wm_momentum)
    crit = nn.CrossEntropyLoss()
    gen = torch.Generator()
    gen.manual_seed(args.seed + 700000 + rnd * 1000 + cid)
    loader = DataLoader(
        wm_dataset, batch_size=args.wm_bs, shuffle=True, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), drop_last=False, generator=gen,
    )
    named = dict(model.named_parameters())
    # Official code creates a fresh optimizer for every client/round and does not explicitly
    # clear gradients between watermark minibatches. Keep this by default for fidelity.
    if args.wm_grad_mode == "zero_each_batch":
        model.zero_grad(set_to_none=True)
    for _ in range(args.wm_epochs):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            # TraMark client i uses MNIST class i and output label i; the original MNIST label is i.
            y = y.to(device, non_blocking=True)
            if args.wm_grad_mode == "zero_each_batch":
                opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            with torch.no_grad():
                for name, p in named.items():
                    if p.grad is not None:
                        p.grad.mul_(wm_mask[name].to(device=p.grad.device, dtype=p.grad.dtype))
            opt.step()
    return cpu_state_dict(model)


def eval_loader(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss_sum += float(crit(logits, y).item())
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return correct / max(total, 1), loss_sum / max(total, 1)


def evaluate_personalized(model_template, states, task_test_loader, wm_test_loaders, device):
    mtas = []
    per_client_wma = []
    for cid in range(len(states)):
        model = copy.deepcopy(model_template)
        load_state(model, states[cid], device)
        task_acc, _ = eval_loader(model, task_test_loader, device)
        wm_acc, _ = eval_loader(model, wm_test_loaders[cid], device)
        mtas.append(task_acc)
        per_client_wma.append(wm_acc)
        model.to("cpu")
        del model
    return {
        "mta": float(np.mean(mtas)),
        "wma": float(np.mean(per_client_wma)),
        "per_client_wma": [float(x) for x in per_client_wma],
    }


def main():
    args = parse_args()
    project = Path(args.project).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Output directory must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    v2 = load_v2_module(project)
    v2.fix_random_seed(args.seed)
    reference_cfg, reference_cfg_path = validate_against_v2_source(project, args)

    # V2 dataset/model args needed by the imported helper functions.
    v2args = SimpleNamespace(
        dataset=args.dataset, data_path=args.data_path, device=device,
        num_clients=args.num_clients, num_outputs=10, num_workers=args.num_workers,
        watermark_source="mnist", wm_train_size=args.wm_train_size,
        wm_test_size=args.wm_test_size, seed=args.seed,
    )
    train_dataset, test_dataset = v2.get_main_datasets(v2args)
    partition, partition_path = exact_v2_partition(project, args.dataset, args.seed)
    client_sets = {cid: Subset(train_dataset, partition[cid]) for cid in range(args.num_clients)}
    sizes = {cid: len(client_sets[cid]) for cid in range(args.num_clients)}

    wm_train_sets, wm_test_sets, wm_meta = build_watermarks(args, v2, v2args, project)
    task_test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), drop_last=False,
    )
    wm_train_loaders = {
        cid: DataLoader(wm_train_sets[cid], batch_size=128, shuffle=False, num_workers=0,
                        pin_memory=(device.type == "cuda"))
        for cid in range(args.num_clients)
    }
    wm_test_loaders = {
        cid: DataLoader(wm_test_sets[cid], batch_size=128, shuffle=False, num_workers=0,
                        pin_memory=(device.type == "cuda"))
        for cid in range(args.num_clients)
    }

    model_template = v2.get_model(args.dataset, 10)
    initial_state = cpu_state_dict(model_template)
    states = {cid: copy.deepcopy(initial_state) for cid in range(args.num_clients)}

    warmup_round = int(args.rounds * args.alpha)
    if warmup_round < 1:
        warmup_round = 1
    main_mask = wm_mask = None

    config = vars(args).copy()
    config.update({
        "device_resolved": str(device),
        "method": "tramark",
        "paper": "Xu et al., ICLR 2026",
        "official_repo": "https://github.com/JiiahaoXU/TraMark",
        "protocol": "FedCAGC V2 fair comparison",
        "non_iid": True,
        "source_client_partition": str(partition_path),
        "source_v2_config": str(reference_cfg_path),
        "warmup_transition_round": warmup_round,
        "wgc_applicable": False,
    })
    save_json(out / "config.json", config)
    save_json(out / "client_partition.json", {str(k): v for k, v in partition.items()})
    save_json(out / "watermark_split.json", wm_meta)

    print("========== TraMark / FedCAGC V2 Fair Baseline ==========")
    for k in ["dataset", "seed", "rounds", "num_clients", "local_epochs", "local_bs", "local_lr", "gamma",
              "wm_train_size", "wm_test_size", "alpha", "k", "wm_epochs", "wm_lr", "wm_bs", "wm_grad_mode", "wm_transform"]:
        print(f"{k:24s}: {getattr(args, k)}", flush=True)
    print(f"warmup_transition_round : {warmup_round}", flush=True)
    print(f"device                  : {device}", flush=True)
    print(f"output_dir              : {out}", flush=True)

    rows = []
    final_eval = None
    total_start = time.perf_counter()

    for rnd in range(1, args.rounds + 1):
        r0 = time.perf_counter()
        print(f"\n-------- Round {rnd:03d}/{args.rounds:03d} --------", flush=True)

        # Client main-task local training from each client's current personalized state.
        local_states = {}
        for cid in range(args.num_clients):
            model = copy.deepcopy(model_template)
            load_state(model, states[cid], device)
            local_states[cid] = train_local_main(model, client_sets[cid], cid, rnd, args, device)
            model.to("cpu")
            del model

        avg_state = fedavg(local_states, sizes)

        if rnd < warmup_round:
            # Standard FedAvg warmup; every client gets the same global state.
            states = {cid: copy.deepcopy(avg_state) for cid in range(args.num_clients)}
            stage = "warmup_fedavg"
        else:
            if rnd == warmup_round:
                # Match the official implementation: mask is initialized from client 0's
                # locally updated model at the warmup transition.
                mask_model = copy.deepcopy(model_template)
                mask_model.load_state_dict(local_states[0], strict=True)
                main_mask, wm_mask = generate_masks(mask_model, args.k)
                total_param = sum(m.numel() for m in wm_mask.values())
                wm_param = sum(int(m.sum().item()) for m in wm_mask.values())
                print(f"TraMark mask initialized: WM parameters={wm_param}/{total_param} ({100*wm_param/total_param:.4f}%)", flush=True)

            states = masked_aggregate(local_states, avg_state, main_mask, wm_mask, model_template)

            # Server-side personalized watermark injection, restricted to watermark region.
            for cid in range(args.num_clients):
                model = copy.deepcopy(model_template)
                load_state(model, states[cid], device)
                states[cid] = inject_watermark(model, wm_train_sets[cid], wm_mask, args, device, cid, rnd)
                model.to("cpu")
                del model
            stage = "tramark"

        round_s = time.perf_counter() - r0

        if should_eval(rnd, args.rounds, args.eval_every, args.eval_tail):
            e0 = time.perf_counter()
            ev = evaluate_personalized(
                model_template, states, task_test_loader, wm_test_loaders, device
            )
            eval_s = time.perf_counter() - e0
            final_eval = ev
            row = {
                "round": rnd,
                "stage": stage,
                "mta": ev["mta"],
                "wma": ev["wma"],
            }
            for cid, x in enumerate(ev["per_client_wma"]):
                row[f"wma_client_{cid}"] = x
            rows.append(row)
            save_rows(out / "round_metrics.csv", rows)
            print(f"MTA                  : {ev['mta']:.4f}", flush=True)
            print(f"WMA                  : {ev['wma']:.4f}", flush=True)
        else:
            print(f"Stage={stage}, round time={round_s:.2f}s (evaluation skipped)", flush=True)

    total_s = time.perf_counter() - total_start
    if final_eval is None:
        raise RuntimeError("No evaluation was produced.")

    summary = {
        "dataset": args.dataset,
        "method": "tramark",
        "seed": args.seed,
        "rounds": args.rounds,
        "num_clients": args.num_clients,
        "final_mta": final_eval["mta"],
        "final_wma": final_eval["wma"],
        "final_wma_per_client": final_eval["per_client_wma"],
        "warmup_transition_round": warmup_round,
        "alpha": args.alpha,
        "k": args.k,
        "wm_lr": args.wm_lr,
        "wm_epochs": args.wm_epochs,
        "wm_grad_mode": args.wm_grad_mode,
        "wm_transform": args.wm_transform,
    }
    save_json(out / "run_summary.json", summary)

    if args.save_checkpoint:
        torch.save({"personalized_states": states, "summary": summary}, out / "final_personalized_models.pt")

    print("\n========== Final Result ==========")
    print(f"Dataset               : {args.dataset}")
    print(f"Method                : TraMark")
    print(f"Final MTA             : {summary['final_mta']:.4f}")
    print(f"Final WMA             : {summary['final_wma']:.4f}")
    print(f"Total seconds         : {summary['total_seconds']:.2f}")
    print(f"Artifacts             : {out}")


if __name__ == "__main__":
    main()
