#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


def read_json(p: Path):
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def load_matrix(p: Path) -> np.ndarray:
    df = pd.read_csv(p, index_col=0)
    return df.to_numpy(dtype=float)


def cosine(a: torch.Tensor, b: torch.Tensor, eps=1e-12) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    if a.numel() != b.numel() or a.numel() == 0:
        return float('nan')
    den = torch.norm(a) * torch.norm(b) + eps
    return float(torch.dot(a, b) / den)


def wgc(grads: List[torch.Tensor], eps=1e-12) -> float:
    if len(grads) < 2:
        return 0.0
    vals = []
    for i in range(len(grads)):
        gi = grads[i].detach().float().reshape(-1)
        ni = torch.norm(gi)
        for j in range(i + 1, len(grads)):
            gj = grads[j].detach().float().reshape(-1)
            c = float(torch.dot(gi, gj) / (ni * torch.norm(gj) + eps))
            vals.append(max(0.0, -c))
    return float(np.mean(vals)) if vals else 0.0


def project_one(g: torch.Tensor, h: torch.Tensor, eps=1e-12) -> torch.Tensor:
    dot = torch.dot(g, h)
    c = dot / (torch.norm(g) * torch.norm(h) + eps)
    if float(c) < 0.0:
        g = g - dot / (torch.dot(h, h) + eps) * h
    return g


def correct_order(g: torch.Tensor, refs: List[torch.Tensor], order: List[int]) -> Tuple[torch.Tensor, float]:
    g0 = g.detach().float().clone().reshape(-1)
    out = g0.clone()
    for idx in order:
        out = project_one(out, refs[idx].detach().float().reshape(-1))
    retain = float(torch.norm(out) / (torch.norm(g0) + 1e-12))
    return out, retain


def cmd_ownership(args):
    root = Path(args.project)
    rows = []
    for ds in args.datasets:
        for seed in args.seeds:
            fed = root / 'results' / 'main' / f'{ds}_fedcagc_{seed}' / 'verification_matrix.csv'
            avg = root / 'results' / 'main' / f'{ds}_fedavg_{seed}' / 'verification_matrix.csv'
            if not fed.exists() or not avg.exists():
                print(f'SKIP missing: {fed} or {avg}')
                continue
            M = load_matrix(fed)
            B = load_matrix(avg)
            mask = ~np.eye(M.shape[0], dtype=bool)
            diag = np.diag(M)
            wrong = M[mask]
            fp_diag = np.diag(B)
            fp_all = B.reshape(-1)
            rows.append({
                'dataset': ds,
                'seed': seed,
                'correct_key_mean': float(diag.mean()),
                'correct_key_min': float(diag.min()),
                'wrong_key_cross_identity_mean': float(wrong.mean()),
                'wrong_key_cross_identity_max': float(wrong.max()),
                'fedavg_false_positive_mean_same_keys': float(fp_diag.mean()),
                'fedavg_false_positive_max_same_keys': float(fp_diag.max()),
                'fedavg_all_key_response_mean': float(fp_all.mean()),
            })
    out = pd.DataFrame(rows)
    outdir = root / 'results' / 'reviewer' / 'offline'
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / 'ownership_wrongkey_fpr.csv', index=False)
    if not out.empty:
        agg = out.groupby('dataset').agg(['mean', 'std'])
        agg.to_csv(outdir / 'ownership_wrongkey_fpr_summary.csv')
    print(out.to_string(index=False))
    print(f'SAVED {outdir / "ownership_wrongkey_fpr.csv"}')


def safe_torch_load(p: Path):
    try:
        return torch.load(p, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(p, map_location='cpu')


def _normalize_snapshot_list(x):
    if x is None:
        return []
    if isinstance(x, dict):
        # client-id keyed dict
        keys = sorted(x, key=lambda z: int(z) if str(z).isdigit() else str(z))
        return [x[k] for k in keys]
    return list(x)


def cmd_snapshot(args):
    root = Path(args.project)
    run = root / 'results' / 'main' / f'{args.dataset}_fedcagc_{args.seed}'
    pp = run / 'prototype_snapshots.pt'
    gp = run / 'gradient_snapshots.pt'
    if not pp.exists() or not gp.exists():
        raise SystemExit(f'Missing snapshot files: {pp} or {gp}')
    protos = safe_torch_load(pp)
    grads = safe_torch_load(gp)
    common = sorted(set(int(k) for k in protos.keys()) & set(int(k) for k in grads.keys()))
    rows = []
    replay_rows = []
    rng_master = np.random.default_rng(args.seed + 20260917)
    for rnd in common:
        P = _normalize_snapshot_list(protos[rnd] if rnd in protos else protos[str(rnd)])
        Gobj = grads[rnd] if rnd in grads else grads[str(rnd)]
        R = _normalize_snapshot_list(Gobj['raw'])
        if len(P) != len(R):
            print(f'SKIP round {rnd}: prototype/gradient client count mismatch {len(P)} vs {len(R)}')
            continue
        cos_own = [cosine(P[i], R[i]) for i in range(len(R)) if P[i] is not None]
        rows.append({
            'round': rnd,
            'clients': len(R),
            'same_round_ema_current_gradient_cos_mean': float(np.nanmean(cos_own)),
            'same_round_ema_current_gradient_cos_std': float(np.nanstd(cos_own, ddof=1)) if len(cos_own) > 1 else 0.0,
            'same_round_ema_current_gradient_cos_min': float(np.nanmin(cos_own)),
            'same_round_ema_current_gradient_cos_max': float(np.nanmax(cos_own)),
        })

        # Offline order replay. Uses the saved round snapshot as a fixed reference bank;
        # this is an order-sensitivity diagnostic, not a re-run of training.
        variants = {}
        retain = {}
        # strength order per client: initial cosine ascending (most negative first)
        corrected = []
        rets = []
        for i, g in enumerate(R):
            refs = [P[j] for j in range(len(P)) if j != i and P[j] is not None]
            sims = [cosine(g, h) for h in refs]
            order = list(np.argsort(sims))
            c, rr = correct_order(g, refs, order)
            corrected.append(c); rets.append(rr)
        variants['strength'] = wgc(corrected); retain['strength'] = float(np.mean(rets))

        corrected = []
        rets = []
        for i, g in enumerate(R):
            refs = [P[j] for j in range(len(P)) if j != i and P[j] is not None]
            order = list(range(len(refs)))
            c, rr = correct_order(g, refs, order)
            corrected.append(c); rets.append(rr)
        variants['fixed_id'] = wgc(corrected); retain['fixed_id'] = float(np.mean(rets))

        random_wgc, random_ret = [], []
        for rep in range(args.random_replays):
            corrected = []; rets = []
            for i, g in enumerate(R):
                refs = [P[j] for j in range(len(P)) if j != i and P[j] is not None]
                order = list(rng_master.permutation(len(refs)))
                c, rr = correct_order(g, refs, order)
                corrected.append(c); rets.append(rr)
            random_wgc.append(wgc(corrected)); random_ret.append(float(np.mean(rets)))
        replay_rows.append({
            'round': rnd,
            'raw_wgc': wgc([x.detach().float().reshape(-1) for x in R]),
            'strength_order_post_wgc': variants['strength'],
            'fixed_id_post_wgc': variants['fixed_id'],
            'random_post_wgc_mean': float(np.mean(random_wgc)),
            'random_post_wgc_std': float(np.std(random_wgc, ddof=1)) if len(random_wgc) > 1 else 0.0,
            'strength_order_norm_retain_mean': retain['strength'],
            'fixed_id_norm_retain_mean': retain['fixed_id'],
            'random_norm_retain_mean': float(np.mean(random_ret)),
            'random_norm_retain_std': float(np.std(random_ret, ddof=1)) if len(random_ret) > 1 else 0.0,
        })

    outdir = root / 'results' / 'reviewer' / 'offline'
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir / f'{args.dataset}_seed{args.seed}_prototype_consistency.csv', index=False)
    pd.DataFrame(replay_rows).to_csv(outdir / f'{args.dataset}_seed{args.seed}_order_replay.csv', index=False)
    print('\nPrototype consistency')
    print(pd.DataFrame(rows).to_string(index=False))
    print('\nOrder replay')
    print(pd.DataFrame(replay_rows).to_string(index=False))


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location('fedcagc_v2_module', str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def auc_manual(y, scores):
    y = np.asarray(y, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos = s[y == 1]; neg = s[y == 0]
    # Mann-Whitney probability of ranking a positive above a negative.
    wins = 0.0
    for a in pos:
        wins += np.sum(a > neg) + 0.5 * np.sum(a == neg)
    return float(wins / (len(pos) * len(neg)))


def best_bal_acc(y, scores):
    y = np.asarray(y, dtype=int); s = np.asarray(scores, dtype=float)
    vals = np.unique(s)
    if len(vals) > 1000:
        vals = np.quantile(s, np.linspace(0, 1, 1001))
    best = (0.0, float('nan'))
    for t in vals:
        pred = (s >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
        tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
        tpr = tp / max(tp + fn, 1); tnr = tn / max(tn + fp, 1)
        ba = 0.5 * (tpr + tnr)
        if ba > best[0]: best = (float(ba), float(t))
    return best


def get_final_linear(model):
    modules = dict(model.named_modules())
    for name in ['classifier.6', 'fc2']:
        if name in modules and isinstance(modules[name], torch.nn.Linear):
            return name, modules[name]
    candidates = [(n, m) for n, m in modules.items() if isinstance(m, torch.nn.Linear)]
    if not candidates:
        raise RuntimeError('No Linear layer found')
    return candidates[-1]


def cmd_leakage(args):
    root = Path(args.project)
    run = root / 'results' / 'main' / f'{args.dataset}_fedcagc_{args.seed}'
    ck = run / 'final_checkpoint.pt'
    splitp = run / 'watermark_split.json'
    cfgp = run / 'config.json'
    codep = root / 'fedcagc_v2_all_methods.py'
    for p in [ck, splitp, cfgp, codep]:
        if not p.exists(): raise SystemExit(f'Missing {p}')
    cfg = read_json(cfgp); split = read_json(splitp)
    mod = load_module(codep)
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = mod.get_model(args.dataset, int(cfg['num_outputs'])).to(device)
    obj = safe_torch_load(ck)
    model.load_state_dict(obj['model_state_dict']); model.eval()
    prototypes = obj.get('prototypes')
    if prototypes is None: raise SystemExit('No prototypes in checkpoint')
    protos = [p.detach().float().to(device).reshape(-1) if p is not None else None for p in prototypes]

    # Build official MNIST train only. Members and nonmembers use the SAME source split
    # to avoid train-vs-test distribution shift confounding membership inference.
    class ArgsObj: pass
    a = ArgsObj()
    for k,v in cfg.items(): setattr(a,k,v)
    a.device = device
    transform = mod.get_mnist_watermark_transform(a)
    train_mnist = mod.datasets.MNIST(cfg['data_path'], train=True, download=True, transform=transform)
    targets = np.asarray(train_mnist.targets)

    lname, final_layer = get_final_linear(model)
    feat_box = {}
    def prehook(module, inp):
        feat_box['h'] = inp[0].detach()
    handle = final_layer.register_forward_pre_hook(prehook)

    all_rows=[]; client_rows=[]
    rng = np.random.default_rng(args.seed + 424242)
    try:
        for cid in range(int(cfg['num_clients'])):
            meta = split['clients'][str(cid)]
            members = [int(x) for x in meta['train_mnist_indices']]
            target_label = int(meta['target_label'])
            source_label = cid  # common V2 MNIST watermark protocol uses label==client id
            candidates = np.where(targets == source_label)[0]
            member_set=set(members)
            nonmembers=[int(x) for x in candidates if int(x) not in member_set]
            rng.shuffle(nonmembers)
            nonmembers=nonmembers[:len(members)]
            proto=protos[cid]
            if proto is None: continue
            C=final_layer.out_features; H=final_layer.in_features
            wlen=C*H
            if proto.numel() != wlen + C:
                raise RuntimeError(f'Prototype length {proto.numel()} != final layer {wlen+C}')
            Pw=proto[:wlen].reshape(C,H); Pb=proto[wlen:]
            pnorm=torch.norm(proto)+1e-12

            def score_indices(indices, member_flag):
                local=[]
                for st in range(0,len(indices),args.batch_size):
                    ids=indices[st:st+args.batch_size]
                    xs=[]
                    for idx in ids:
                        x,_=train_mnist[idx]; xs.append(x)
                    x=torch.stack(xs).to(device)
                    with torch.no_grad():
                        logits=model(x); h=feat_box['h']
                        prob=torch.softmax(logits,dim=1)
                        delta=prob.clone(); delta[:,target_label]-=1.0
                        # <grad_i, proto> = delta_i dot (Pw*h_i + Pb)
                        projected=torch.nn.functional.linear(h,Pw,Pb)
                        dot=(delta*projected).sum(dim=1)
                        gnorm=torch.norm(delta,dim=1)*torch.sqrt((h*h).sum(dim=1)+1.0)
                        scores=(dot/(gnorm*pnorm+1e-12)).detach().cpu().numpy()
                    for idx,sc in zip(ids,scores):
                        all_rows.append({'client':cid,'index':idx,'member':member_flag,'score':float(sc)})
                        local.append(float(sc))
                return local
            sm=score_indices(members,1); sn=score_indices(nonmembers,0)
            y=np.array([1]*len(sm)+[0]*len(sn)); sc=np.array(sm+sn)
            auc=auc_manual(y,sc); ba,thr=best_bal_acc(y,sc)
            client_rows.append({'client':cid,'members':len(sm),'nonmembers':len(sn),'auc':auc,'best_balanced_accuracy':ba,'best_threshold':thr,
                                'member_score_mean':float(np.mean(sm)),'nonmember_score_mean':float(np.mean(sn))})
    finally:
        handle.remove()

    outdir=root/'results'/'reviewer'/'offline'; outdir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(all_rows).to_csv(outdir/f'{args.dataset}_seed{args.seed}_prototype_leakage_scores.csv',index=False)
    cdf=pd.DataFrame(client_rows); cdf.to_csv(outdir/f'{args.dataset}_seed{args.seed}_prototype_leakage_summary.csv',index=False)
    if all_rows:
        y=np.array([r['member'] for r in all_rows]); sc=np.array([r['score'] for r in all_rows])
        auc=auc_manual(y,sc); ba,thr=best_bal_acc(y,sc)
        overall={'dataset':args.dataset,'seed':args.seed,'auc':auc,'best_balanced_accuracy':ba,'best_threshold':thr,
                 'mean_client_auc':float(cdf.auc.mean()),'std_client_auc':float(cdf.auc.std(ddof=1))}
        with (outdir/f'{args.dataset}_seed{args.seed}_prototype_leakage_overall.json').open('w') as f: json.dump(overall,f,indent=2)
        print(json.dumps(overall,indent=2))
        print(cdf.to_string(index=False))


def cmd_stats(args):
    root=Path(args.project)
    rows=[]
    for ds in args.datasets:
        for method in args.methods:
            for seed in args.seeds:
                run=root/'results'/'main'/f'{ds}_{method}_{seed}'
                sp=run/'run_summary.json'; cp=run/'round_metrics.csv'
                if not sp.exists(): continue
                s=read_json(sp)
                row={'dataset':ds,'method':method,'seed':seed,
                     'MTA':s.get('final_mta'),'WMA':s.get('final_wma'),'MinWMA':s.get('final_min_wma'),
                     'ClientWMA_SD':s.get('final_std_wma'),
                     'WGC_pre':s.get('final_wgc_pre'),'WGC_post':s.get('final_wgc_post')}
                if cp.exists():
                    rdf=pd.read_csv(cp)
                    rr=rdf[rdf['round']==100]
                    if len(rr):
                        rec=rr.iloc[-1]
                        vals=[float(rec[f'wma_client_{i}']) for i in range(10) if f'wma_client_{i}' in rdf.columns]
                        if vals:
                            row['ClientWMA_Gap']=max(vals)-min(vals)
                            row['ClientWMA_Min']=min(vals)
                rows.append(row)
    df=pd.DataFrame(rows)
    outdir=root/'results'/'reviewer'/'offline'; outdir.mkdir(parents=True,exist_ok=True)
    df.to_csv(outdir/'main_seed_metrics.csv',index=False)
    summaries=[]
    if not df.empty:
        for (ds,m),g in df.groupby(['dataset','method']):
            r={'dataset':ds,'method':m,'n':len(g)}
            for col in ['MTA','WMA','MinWMA','ClientWMA_SD','ClientWMA_Gap','ClientWMA_Min','WGC_pre','WGC_post']:
                x=g[col].dropna().to_numpy(float)
                if len(x):
                    mean=float(x.mean()); sd=float(x.std(ddof=1)) if len(x)>1 else 0.0
                    # t-based 95% CI for n=3 without scipy; t_0.975,df=2 = 4.30265
                    tcrit=4.302652729911275 if len(x)==3 else 1.96
                    ci=tcrit*sd/math.sqrt(len(x)) if len(x)>1 else 0.0
                    r[col+'_mean']=mean; r[col+'_sd']=sd; r[col+'_ci95_halfwidth']=ci
            summaries.append(r)
    sdf=pd.DataFrame(summaries); sdf.to_csv(outdir/'main_mean_sd_ci95.csv',index=False)
    print(sdf.to_string(index=False))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',default='/home/test/lxl/FedCAGC_V2')
    sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('ownership'); p.add_argument('--datasets',nargs='+',default=['cifar10','fmnist']); p.add_argument('--seeds',nargs='+',type=int,default=[3047,3048,3049]); p.set_defaults(func=cmd_ownership)
    p=sub.add_parser('snapshot'); p.add_argument('--dataset',default='cifar10'); p.add_argument('--seed',type=int,default=3047); p.add_argument('--random_replays',type=int,default=20); p.set_defaults(func=cmd_snapshot)
    p=sub.add_parser('leakage'); p.add_argument('--dataset',default='cifar10'); p.add_argument('--seed',type=int,default=3047); p.add_argument('--batch_size',type=int,default=64); p.add_argument('--device',default=None); p.set_defaults(func=cmd_leakage)
    p=sub.add_parser('stats'); p.add_argument('--datasets',nargs='+',default=['cifar10','fmnist']); p.add_argument('--methods',nargs='+',default=['fedavg','fedipr','flwb','fedawm','fedcagc']); p.add_argument('--seeds',nargs='+',type=int,default=[3047,3048,3049]); p.set_defaults(func=cmd_stats)
    args=ap.parse_args(); args.func(args)

if __name__=='__main__': main()
