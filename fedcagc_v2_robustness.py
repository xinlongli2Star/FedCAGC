#!/usr/bin/env python3
"""V2 post-training robustness evaluation for FedCAGC.

This script does NOT retrain federated learning. It loads an already-trained V2
FedCAGC final checkpoint and evaluates two post-processing attacks:

1) Clean fine-tuning attack:
   - Fine-tune only on the official clean main-task training set.
   - No watermark samples are used during fine-tuning.
   - Default: 50 epochs; evaluate MTA/WMA at epoch 0 and after every epoch.

2) Global unstructured magnitude pruning attack:
   - Reload the ORIGINAL final checkpoint independently for every pruning ratio.
   - Globally prune Conv2d/Linear weight tensors by absolute magnitude.
   - Biases are not pruned; no post-pruning recovery/fine-tuning is performed.
   - Default ratios: 0,10,20,40,60,80,90,95,99%.

The watermark test set is reconstructed by importing fedcagc_v2_all_methods.py
from the same project directory, ensuring the exact V2 watermark protocol:
100/client official MNIST train for embedding and 200/client official MNIST test
for verification, with train/test strictly separated by the official split.
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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser(description="FedCAGC V2 fine-tuning/pruning robustness evaluation")
    p.add_argument("--project", type=str, default="/home/test/lxl/FedCAGC_V2")
    p.add_argument("--dataset", choices=["fmnist", "cifar10"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--ft_epochs", type=int, default=50)
    p.add_argument("--ft_lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip_norm", type=float, default=20.0)
    p.add_argument("--prune_ratios", type=str, default="0,10,20,40,60,80,90,95,99")
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def load_v2_module(project: Path):
    src = project / "fedcagc_v2_all_methods.py"
    if not src.exists():
        raise FileNotFoundError(f"Missing V2 runner: {src}")
    spec = importlib.util.spec_from_file_location("fedcagc_v2_all_methods", str(src))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_csv(path: Path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_eval_args(base_cfg, args):
    cfg = dict(base_cfg)
    cfg["data_path"] = str(Path(cfg.get("data_path", "/home/test/lxl/data")).expanduser())
    cfg["device"] = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    cfg["num_workers"] = args.num_workers
    # V2 formal protocol is fixed, but preserve recorded config values if present.
    cfg.setdefault("watermark_source", "mnist")
    cfg.setdefault("wm_train_size", 100)
    cfg.setdefault("wm_test_size", 200)
    cfg.setdefault("num_clients", 10)
    cfg.setdefault("num_outputs", 10)
    return SimpleNamespace(**cfg)


def load_model(mod, run_dir: Path, eval_args):
    ckpt_path = run_dir / "final_checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model = mod.get_model(eval_args.dataset, int(eval_args.num_outputs))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(eval_args.device)
    return model, ckpt


def eval_all(mod, model, task_test, wm_test_sets, target_labels, eval_args):
    mta, task_loss = mod.evaluate_task(model, task_test, eval_args)
    per_client, wma, min_wma, wm_loss, cross = mod.evaluate_watermarks(
        model, wm_test_sets, target_labels, eval_args
    )
    return {
        "mta": float(mta),
        "wma": float(wma),
        "min_wma": float(min_wma),
        "std_wma": float(np.std(per_client, ddof=0)),
        "task_loss": float(task_loss),
        "wm_loss": float(wm_loss),
        "cross_identity_response": float(cross),
        "per_client_wma": [float(x) for x in per_client],
    }


def fine_tune_attack(mod, base_model, task_train, task_test, wm_test_sets, target_labels, eval_args, args, out_dir):
    model = copy.deepcopy(base_model).to(eval_args.device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.ft_lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    gen = torch.Generator()
    gen.manual_seed(args.seed + 20260921)
    loader = DataLoader(
        task_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(eval_args.device.type == "cuda"),
        drop_last=False,
        generator=gen,
    )

    rows = []
    start = eval_all(mod, model, task_test, wm_test_sets, target_labels, eval_args)
    row = {"epoch": 0, **{k: v for k, v in start.items() if k != "per_client_wma"}}
    for i, x in enumerate(start["per_client_wma"]):
        row[f"wma_client_{i}"] = x
    rows.append(row)
    print(f"[FineTune] epoch=000 MTA={start['mta']:.4f} WMA={start['wma']:.4f} MinWMA={start['min_wma']:.4f}", flush=True)

    t0 = time.perf_counter()
    for epoch in range(1, args.ft_epochs + 1):
        model.train()
        loss_sum = 0.0
        n_seen = 0
        for x, y in loader:
            x = x.to(eval_args.device, non_blocking=True)
            y = y.to(eval_args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            loss_sum += float(loss.item()) * int(y.size(0))
            n_seen += int(y.size(0))

        metrics = eval_all(mod, model, task_test, wm_test_sets, target_labels, eval_args)
        row = {
            "epoch": epoch,
            "train_clean_loss": loss_sum / max(1, n_seen),
            **{k: v for k, v in metrics.items() if k != "per_client_wma"},
        }
        for i, x in enumerate(metrics["per_client_wma"]):
            row[f"wma_client_{i}"] = x
        rows.append(row)
        print(
            f"[FineTune] epoch={epoch:03d} MTA={metrics['mta']:.4f} "
            f"WMA={metrics['wma']:.4f} MinWMA={metrics['min_wma']:.4f}",
            flush=True,
        )

    total_s = time.perf_counter() - t0
    fields = ["epoch", "train_clean_loss", "mta", "wma", "min_wma", "std_wma", "task_loss", "wm_loss", "cross_identity_response"]
    fields += [f"wma_client_{i}" for i in range(int(eval_args.num_clients))]
    save_csv(out_dir / "finetune_metrics.csv", rows, fields)

    wmas = np.asarray([r["wma"] for r in rows], dtype=np.float64)
    mtas = np.asarray([r["mta"] for r in rows], dtype=np.float64)
    summary = {
        "start_mta": rows[0]["mta"],
        "start_wma": rows[0]["wma"],
        "final_mta": rows[-1]["mta"],
        "final_wma": rows[-1]["wma"],
        "wma_drop_final_percentage_points": (rows[0]["wma"] - rows[-1]["wma"]) * 100.0,
        "minimum_wma": float(wmas.min()),
        "minimum_wma_epoch": int(rows[int(wmas.argmin())]["epoch"]),
        "maximum_mta": float(mtas.max()),
        "epochs": args.ft_epochs,
        "clean_only": True,
        "watermark_samples_used_for_finetuning": False,
        "optimizer": "SGD",
        "lr": args.ft_lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip_norm,
        "total_seconds_excluding_epoch0_eval": total_s,
    }
    save_json(out_dir / "finetune_summary.json", summary)
    return summary


def prunable_weight_tensors(model):
    out = []
    for module_name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            out.append((f"{module_name}.weight", module.weight))
    return out


def apply_global_magnitude_pruning(model, ratio):
    """Set exactly floor(ratio * N) smallest-magnitude Conv/Linear weights to zero."""
    tensors = prunable_weight_tensors(model)
    if not tensors:
        raise RuntimeError("No Conv2d/Linear weights found for pruning.")
    flat_abs = torch.cat([p.detach().abs().reshape(-1).cpu() for _, p in tensors], dim=0)
    total = int(flat_abs.numel())
    k = int(math.floor(float(ratio) * total))
    if k <= 0:
        return 0, total
    if k >= total:
        threshold = float("inf")
    else:
        # kthvalue is 1-indexed. We use a stable global ranking below to guarantee exact k.
        threshold = float(torch.kthvalue(flat_abs, k).values.item())

    # First zero everything strictly below threshold; then resolve ties deterministically.
    below_counts = []
    equal_counts = []
    for _, p in tensors:
        a = p.detach().abs()
        below_counts.append(int((a < threshold).sum().item()))
        equal_counts.append(int((a == threshold).sum().item()))
    below_total = sum(below_counts)
    remaining = max(0, k - below_total)

    with torch.no_grad():
        # deterministic traversal by module order and flattened parameter index
        for _, p in tensors:
            a = p.detach().abs()
            mask = a < threshold
            p[mask] = 0.0
        if remaining > 0 and math.isfinite(threshold):
            for _, p in tensors:
                if remaining <= 0:
                    break
                a = p.detach().abs()
                eq_idx = torch.nonzero(a.reshape(-1) == threshold, as_tuple=False).reshape(-1)
                if eq_idx.numel() == 0:
                    continue
                take = min(remaining, int(eq_idx.numel()))
                flat = p.view(-1)
                flat[eq_idx[:take]] = 0.0
                remaining -= take
        elif not math.isfinite(threshold):
            for _, p in tensors:
                p.zero_()

    zeros = sum(int((p.detach() == 0).sum().item()) for _, p in tensors)
    return zeros, total


def pruning_attack(mod, base_model, task_test, wm_test_sets, target_labels, eval_args, args, out_dir):
    ratios_pct = [float(x.strip()) for x in args.prune_ratios.split(",") if x.strip()]
    rows = []
    for pct in ratios_pct:
        if not (0 <= pct <= 100):
            raise ValueError(f"Invalid pruning ratio: {pct}")
        # IMPORTANT: reload the original unpruned model independently for each ratio.
        model = copy.deepcopy(base_model).to(eval_args.device)
        zeros, total = apply_global_magnitude_pruning(model, pct / 100.0)
        metrics = eval_all(mod, model, task_test, wm_test_sets, target_labels, eval_args)
        row = {
            "prune_ratio_percent": pct,
            "pruned_or_zero_weight_count": zeros,
            "prunable_weight_count": total,
            "measured_zero_fraction": zeros / max(1, total),
            **{k: v for k, v in metrics.items() if k != "per_client_wma"},
        }
        for i, x in enumerate(metrics["per_client_wma"]):
            row[f"wma_client_{i}"] = x
        rows.append(row)
        print(
            f"[Prune] ratio={pct:5.1f}% zero={100.0*zeros/max(1,total):6.2f}% "
            f"MTA={metrics['mta']:.4f} WMA={metrics['wma']:.4f} MinWMA={metrics['min_wma']:.4f}",
            flush=True,
        )

    fields = ["prune_ratio_percent", "pruned_or_zero_weight_count", "prunable_weight_count", "measured_zero_fraction",
              "mta", "wma", "min_wma", "std_wma", "task_loss", "wm_loss", "cross_identity_response"]
    fields += [f"wma_client_{i}" for i in range(int(eval_args.num_clients))]
    save_csv(out_dir / "pruning_metrics.csv", rows, fields)
    save_json(out_dir / "pruning_summary.json", {
        "ratios_percent": ratios_pct,
        "pruning_type": "global unstructured magnitude pruning",
        "pruned_parameters": "Conv2d/Linear weight tensors only; biases unchanged",
        "independent_from_original_checkpoint": True,
        "post_pruning_finetune": False,
        "results": rows,
    })
    return rows


def main():
    args = parse_args()
    project = Path(args.project).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = project / "results" / "main" / f"{args.dataset}_fedcagc_{args.seed}"
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing V2 config: {cfg_path}")
    base_cfg = load_json(cfg_path)
    if int(base_cfg.get("rounds", -1)) != 100:
        raise RuntimeError(f"Refusing non-V2 checkpoint: rounds={base_cfg.get('rounds')}, expected 100")
    if int(base_cfg.get("seed", -1)) != args.seed or base_cfg.get("dataset") != args.dataset:
        raise RuntimeError("Checkpoint config does not match requested dataset/seed")
    if base_cfg.get("method") != "fedcagc":
        raise RuntimeError("Robustness script expects a FedCAGC V2 checkpoint")

    mod = load_v2_module(project)
    mod.fix_random_seed(args.seed)
    eval_args = build_eval_args(base_cfg, args)

    print("========== FedCAGC V2 Robustness ==========")
    print("dataset       :", args.dataset)
    print("seed          :", args.seed)
    print("checkpoint    :", run_dir / "final_checkpoint.pt")
    print("device        :", eval_args.device)
    print("ft_epochs     :", args.ft_epochs)
    print("prune_ratios  :", args.prune_ratios)
    print("output_dir    :", out_dir)

    task_train, task_test = mod.get_main_datasets(eval_args)
    wm_train_sets, wm_extra_sets, wm_test_sets, target_labels, wm_metadata = mod.build_watermark_datasets(eval_args)
    base_model, ckpt = load_model(mod, run_dir, eval_args)

    # Verify that the reconstructed V2 test protocol reproduces the checkpoint's starting point.
    start = eval_all(mod, base_model, task_test, wm_test_sets, target_labels, eval_args)
    recorded = ckpt.get("final_record", {})
    print(
        f"[StartCheck] reconstructed MTA={start['mta']:.4f}, WMA={start['wma']:.4f}; "
        f"checkpoint-record MTA={float(recorded.get('mta', float('nan'))):.4f}, "
        f"WMA={float(recorded.get('wma', float('nan'))):.4f}",
        flush=True,
    )
    if recorded:
        rec_mta = float(recorded.get("mta", start["mta"]))
        rec_wma = float(recorded.get("wma", start["wma"]))
        # GPU evaluation can differ by one or two borderline samples across fresh processes
        # because convolution/TF32 kernels are not guaranteed to be bitwise identical.
        # Use a sample-granularity tolerance only for this protocol-consistency guard;
        # the actual reported robustness metrics are never rounded or altered.
        mta_tol = max(2.0 / max(1, len(task_test)), 1e-6)
        wm_total = sum(len(ds) for ds in wm_test_sets)
        wma_tol = max(2.0 / max(1, wm_total), 1e-6)
        mta_diff = abs(start["mta"] - rec_mta)
        wma_diff = abs(start["wma"] - rec_wma)
        print(
            f"[StartCheck] abs diff: MTA={mta_diff:.8f} (tol={mta_tol:.8f}), "
            f"WMA={wma_diff:.8f} (tol={wma_tol:.8f})",
            flush=True,
        )
        if mta_diff > mta_tol or wma_diff > wma_tol:
            raise RuntimeError(
                "Reconstructed V2 evaluation differs beyond sample-granularity tolerance "
                "from checkpoint final_record; stop to avoid protocol mismatch."
            )
        if mta_diff > 1e-6 or wma_diff > 1e-6:
            print(
                "[StartCheck] PASS with sample-level tolerance; this is treated as GPU numerical "
                "non-bitwise reproducibility, not a protocol mismatch.",
                flush=True,
            )

    save_json(out_dir / "robustness_config.json", {
        "dataset": args.dataset,
        "seed": args.seed,
        "source_run_dir": str(run_dir),
        "source_rounds": base_cfg["rounds"],
        "source_watermark_protocol": {
            "source": base_cfg.get("watermark_source"),
            "wm_train_size": base_cfg.get("wm_train_size"),
            "wm_test_size": base_cfg.get("wm_test_size"),
            "official_train_test_separation": True,
        },
        "fine_tuning": {
            "epochs": args.ft_epochs,
            "batch_size": args.batch_size,
            "lr": args.ft_lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "grad_clip_norm": args.grad_clip_norm,
            "clean_main_task_training_data_only": True,
            "watermark_samples_used": False,
        },
        "pruning": {
            "ratios_percent": [float(x.strip()) for x in args.prune_ratios.split(",") if x.strip()],
            "type": "global unstructured magnitude pruning",
            "scope": "Conv2d/Linear weights",
            "reload_original_checkpoint_for_each_ratio": True,
            "post_pruning_finetune": False,
        },
    })

    ft_summary = fine_tune_attack(
        mod, base_model, task_train, task_test, wm_test_sets, target_labels, eval_args, args, out_dir
    )
    prune_rows = pruning_attack(
        mod, base_model, task_test, wm_test_sets, target_labels, eval_args, args, out_dir
    )

    save_json(out_dir / "run_summary.json", {
        "dataset": args.dataset,
        "seed": args.seed,
        "initial": start,
        "finetune": ft_summary,
        "pruning_final_points": [
            {"ratio_percent": r["prune_ratio_percent"], "mta": r["mta"], "wma": r["wma"]}
            for r in prune_rows
        ],
    })
    print("DONE:", out_dir, flush=True)


if __name__ == "__main__":
    main()
