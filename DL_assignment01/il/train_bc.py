#!/usr/bin/env python3
"""Milestone 3 (part 3): behaviour cloning on your demonstrations.

    python il/train_bc.py --data demos/20261234 --out runs/m3            # after the TODOs in dataset.py and policy.py
    python il/train_bc.py --data demos/20261234 --out runs/m3 --dry-run  # one batch forward + backward, no training

Given: argument parsing, data loading and the split, the checkpoint and the log format.
TODO:  the training loop (marked below).
Writes OUT/ckpt.pt (policy.load_agent reads it), OUT/train_log.csv (epoch, train_loss, val_loss), OUT/config.json.
"""
import argparse, csv, json, time
from pathlib import Path

import numpy as np
import torch

from dataset import load_demos, split_by_episode, make_pairs, Normalizer, BCDataset
from policy import MLPPolicy, save_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', nargs='+', required=True, help='demo folders (ep_*.npz)')
    ap.add_argument('--out', type=Path, default=Path('runs/m3'))
    ap.add_argument('--epochs', type=int, default=200); ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--hidden', type=int, default=256); ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--obs-horizon', type=int, default=1); ap.add_argument('--act-horizon', type=int, default=1)
    ap.add_argument('--only-success', action='store_true'); ap.add_argument('--keep-idle', action='store_true')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    # ---- data (given)
    eps = load_demos(a.data, drop_idle=not a.keep_idle, only_success=a.only_success)
    tr, va = split_by_episode(eps, .2, a.seed)
    print(f'{len(eps)} episodes ({sum(e["length"] for e in eps)} steps): {len(tr)} train, {len(va)} val')
    Xtr, Ytr = make_pairs(tr, a.obs_horizon, a.act_horizon); Xva, Yva = make_pairs(va, a.obs_horizon, a.act_horizon)
    x_norm, y_norm = Normalizer(), Normalizer(); x_norm.fit(Xtr); y_norm.fit(Ytr)
    dl_tr = torch.utils.data.DataLoader(BCDataset(Xtr, Ytr, x_norm, y_norm), batch_size=a.batch, shuffle=True)
    dl_va = torch.utils.data.DataLoader(BCDataset(Xva, Yva, x_norm, y_norm), batch_size=a.batch)
    config = dict(model=dict(in_dim=Xtr.shape[1], out_dim=Ytr.shape[1], hidden=a.hidden, layers=a.layers),
                  obs_horizon=a.obs_horizon, act_horizon=a.act_horizon, data=a.data, n_train=len(Xtr), n_val=len(Xva),
                  train_episodes=[e['name'] for e in tr], val_episodes=[e['name'] for e in va], args={k: str(v) for k, v in vars(a).items()})
    (a.out / 'config.json').write_text(json.dumps(config, indent=2))
    model = MLPPolicy(**config['model'])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    print(f'X {Xtr.shape}, Y {Ytr.shape}; {sum(p.numel() for p in model.parameters())} parameters')

    if a.dry_run:      # one batch, to check shapes and gradients before training
        x, y = next(iter(dl_tr)); loss = torch.nn.functional.mse_loss(model(x), y); loss.backward(); opt.step()
        print(f'dry run: batch {tuple(x.shape)} -> {tuple(model(x).shape)}, loss {loss.item():.4f}; OK'); return

    # ---- training (TODO)
    log = open(a.out / 'train_log.csv', 'w', newline=''); w = csv.writer(log); w.writerow(['epoch', 'train_loss', 'val_loss', 'time_s'])
    best, t0 = float('inf'), time.time()
    for epoch in range(a.epochs):
        # TODO 1: one pass over dl_tr: model.train(), predict, MSE loss on the normalised actions, backward, opt.step().
        #         Keep the mean training loss in train_loss.
        # TODO 2: the validation loss on dl_va under torch.no_grad() with model.eval(), in val_loss.
        # TODO 3: whenever val_loss improves on best, save_checkpoint(a.out / 'ckpt.pt', model, config, x_norm, y_norm,
        #         epoch=epoch, val_loss=val_loss).
        # Optional: a learning-rate schedule, early stopping, an L1 / Huber loss (say why in the report).
        raise NotImplementedError('TODO: il/train_bc.py training loop')
        w.writerow([epoch, train_loss, val_loss, round(time.time() - t0, 1)]); log.flush()
    log.close()


if __name__ == '__main__':
    main()
