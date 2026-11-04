#!/usr/bin/env python3
"""Milestone 2: check and summarise a folder of demonstrations. Validates every episode (sim/episode_io.problems),
prints one line each and the totals, and draws them: each episode's stick-tip path over its start and final rope, with
the goal T.

    python scripts/inspect_demos.py demos/20261234
Writes runs/m2/summary.csv, runs/m2/summary.json and runs/m2/demos.png (deliverables of Milestone 2).
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sim'))
BRANCHES = (range(0, 5), range(5, 10), range(10, 19))


def plot(eps, path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    n = len(eps); c = min(n, 5); r = (n + c - 1) // c
    fig, axs = plt.subplots(r, c, figsize=(3.2 * c, 3.2 * r), squeeze=False)
    for ax, (name, ep) in zip(axs.ravel(), eps):
        for kp, col, lab in ((ep['goal'], 'tab:green', 'goal'), (ep['kp'][0], 'tab:gray', 'start'), (ep['kp'][-1], 'tab:red', 'end')):
            for i, br in enumerate(BRANCHES): ax.plot(*kp[list(br)].T, '-o', ms=2, color=col, label=lab if i == 0 else None)
        ax.plot(*ep['tip'][:, :2].T, color='tab:blue', lw=.8, label='stick tip')
        e = ep['err'][-1]; ax.set_title(f'{name}\n{"OK" if ep["success"] else "--"} pos {e[0]:.0f} rot {e[1]:.0f} shape {e[2]:.0f}', fontsize=8)
        ax.set_aspect('equal'); ax.tick_params(labelsize=6)
    for ax in axs.ravel()[n:]: ax.axis('off')
    axs[0, 0].legend(fontsize=6); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder', type=Path); ap.add_argument('--out', type=Path, default=ROOT / 'runs' / 'm2')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    from episode_io import load, problems
    files = sorted(a.folder.glob('ep_*.npz'))
    if not files: sys.exit(f'no ep_*.npz in {a.folder}')
    rows, good = [], []
    print(f'{"file":>16} {"steps":>5} {"driven":>6} {"success":>7} {"pos mm":>6} {"rot deg":>7} {"shape mm":>8}  problems')
    for f in files:
        ep = load(f); bad = problems(ep)
        if bad: rows.append(dict(file=f.name, valid=False, problems='; '.join(bad))); print(f'{f.name:>16}  INVALID: {"; ".join(bad)}'); continue
        e = ep['err'][-1]; good.append((f.stem, ep))
        rows.append(dict(file=f.name, valid=True, seed=ep['seed'], student=ep['student'], steps=ep['length'], driven=int(ep['human'].sum()),
                         success=bool(ep['success']), ever_success=bool(ep['succ'].any()), pos_mm=round(float(e[0]), 1), rot_deg=round(float(e[1]), 1), shape_mm=round(float(e[2]), 1), problems=''))
        r = rows[-1]; print(f'{f.name:>16} {r["steps"]:5d} {r["driven"]:6d} {str(r["success"]):>7} {r["pos_mm"]:6.1f} {r["rot_deg"]:7.1f} {r["shape_mm"]:8.1f}')
    with open(a.out / 'summary.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r))); w.writeheader(); w.writerows(rows)
    ok = [r for r in rows if r['valid']]
    tot = dict(folder=str(a.folder), episodes=len(rows), valid=len(ok), success=sum(r['success'] for r in ok),
               steps=sum(r['steps'] for r in ok), driven_steps=sum(r['driven'] for r in ok),
               seeds=sorted(int(r['seed']) for r in ok), students=sorted({r['student'] for r in ok}))
    (a.out / 'summary.json').write_text(json.dumps(tot, indent=2))
    if good: plot(good, a.out / 'demos.png')
    print(f'\n{tot["valid"]}/{tot["episodes"]} valid, {tot["success"]} successful, {tot["driven_steps"]} driven steps of {tot["steps"]} -> {a.out}')


if __name__ == '__main__':
    main()
