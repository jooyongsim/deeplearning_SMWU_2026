#!/usr/bin/env python3
"""Milestone 4: run a policy in the simulator on the shared held-out starts (scene.EVAL_SEEDS, no training uses them)
and score it. Given in full.

    python rl/eval_policy.py --agent random                              # baseline: random keypoint folds
    python rl/eval_policy.py --agent ppo --ckpt runs/m3/ppo/ckpt.zip      # your policy, all 50 starts
    python rl/eval_policy.py --agent ppo --ckpt runs/m3/ppo/ckpt.zip --n 3 --video 3
    python rl/eval_policy.py --agent ppo --ckpt ... --mesh assets/my_cloth --out runs/m4/ppo_mycloth

Success = sim/fold_metrics.py (largest corner gap <= 70 mm and covered area <= 0.35) after the 2 folds.
Writes OUT/eval.json and OUT/rollouts/ep_<seed>.npz (sim/episode_io.py format; scripts/render_video.py makes mp4s).
--workers runs that many simulators in parallel (default: the CPU count, at most 8).
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sim')); sys.path.insert(0, str(ROOT / 'rl')); sys.path.insert(0, str(ROOT))


def run(job):
    """One evaluation episode in a worker process."""
    sys.path.insert(0, str(ROOT / 'sim')); sys.path.insert(0, str(ROOT / 'rl'))
    seed, a = job
    from fold_env import FoldEnv
    from episode_io import Recorder
    import torch; torch.set_num_threads(1)
    env = FoldEnv(mesh=a['mesh'], action=a['action'], record=lambda e, s: Recorder(e, s, a['mesh'] or 'cloth_example', a['agent'], f'eval:{a["agent"]}'))
    if a['agent'] == 'ppo':
        from stable_baselines3 import PPO
        model = PPO.load(a['ckpt'], device='cpu')
    rng = np.random.default_rng(seed); t0 = time.time()
    obs, info = env.reset(seed=seed, options=dict(env_seed=seed)); ret, done, picks = 0., False, []
    while not done:
        act = model.predict(obs, deterministic=True)[0] if a['agent'] == 'ppo' else env.action_space.sample() if a['agent'] == 'random' else None
        obs, r, done, _, info = env.step(act); ret += r; picks.append(np.asarray(act).tolist())
    env.rec.save(Path(a['out']) / 'rollouts' / f'ep_{seed}.npz')
    return dict(seed=seed, success=bool(info['success']), gap_a=round(info['gap_a'], 1), gap_b=round(info['gap_b'], 1), cover=round(info['cover'], 3),
                folds=info['folds'], ret=round(ret, 3), actions=picks, steps=len(env.rec), wall_s=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--agent', choices=['random', 'ppo'], required=True); ap.add_argument('--ckpt', type=Path)
    ap.add_argument('--action', choices=['keypoint', 'xy'], default=None, help='default: from the checkpoint folder config.json, else keypoint')
    ap.add_argument('--mesh', default=None); ap.add_argument('--n', type=int, default=50, help='first n of the 50 eval seeds')
    ap.add_argument('--workers', type=int, default=min(os.cpu_count() or 1, 8)); ap.add_argument('--video', type=int, default=0, help='render the first k episodes')
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    if a.agent == 'ppo' and not a.ckpt: ap.error('--ckpt is needed for --agent ppo')
    if a.action is None:
        cfg = a.ckpt.parent / 'config.json' if a.ckpt else None
        a.action = json.loads(cfg.read_text()).get('action', 'keypoint') if cfg and cfg.exists() else 'keypoint'
    a.out = a.out or ROOT / 'runs' / 'm4' / a.agent; (a.out / 'rollouts').mkdir(parents=True, exist_ok=True)
    from scene import EVAL_SEEDS
    seeds = list(EVAL_SEEDS)[:a.n]
    job = dict(agent=a.agent, ckpt=str(a.ckpt) if a.ckpt else None, mesh=a.mesh, action=a.action, out=str(a.out))
    t0 = time.time(); rows = []
    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(a.workers) as ex:
            for r in ex.map(run, [(s, job) for s in seeds]): rows.append(r); print(f'  seed {r["seed"]}: success {r["success"]}  gaps {r["gap_a"]:.0f} / {r["gap_b"]:.0f} mm  cover {r["cover"]:.2f}', flush=True)
    else:
        for s in seeds: r = run((s, job)); rows.append(r); print(f'  seed {s}: success {r["success"]}  gaps {r["gap_a"]:.0f} / {r["gap_b"]:.0f} mm  cover {r["cover"]:.2f}', flush=True)
    g = np.array([max(r['gap_a'], r['gap_b']) for r in rows])
    rep = dict(agent=a.agent, ckpt=str(a.ckpt) if a.ckpt else None, action=a.action, mesh=a.mesh or 'assets/cloth_example', n=len(rows),
               success_rate=float(np.mean([r['success'] for r in rows])), two_folds_rate=float(np.mean([r['folds'] == 2 for r in rows])),
               max_gap_median_mm=float(np.median(g)), cover_median=float(np.median([r['cover'] for r in rows])),
               mean_return=float(np.mean([r['ret'] for r in rows])), time=time.strftime('%Y-%m-%d %H:%M:%S'), wall_s=round(time.time() - t0), episodes=rows)
    (a.out / 'eval.json').write_text(json.dumps(rep, indent=2))
    print(f'{a.agent}: success {rep["success_rate"] * 100:.0f} % ({len(rows)} seeds), largest corner gap median {rep["max_gap_median_mm"]:.0f} mm, '
          f'cover median {rep["cover_median"]:.2f}  -> {a.out / "eval.json"}')
    if a.video:
        from scripts.render_video import render
        from episode_io import load
        (a.out / 'videos').mkdir(exist_ok=True)
        for s in seeds[:a.video]:
            p = a.out / 'videos' / f'ep_{s}.mp4'; render(load(a.out / 'rollouts' / f'ep_{s}.npz'), p); print(f'  video -> {p}')


if __name__ == '__main__':
    main()
