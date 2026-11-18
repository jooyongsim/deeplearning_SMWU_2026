#!/usr/bin/env python3
"""Milestone 3: train a folding policy with PPO (stable-baselines3), locally or on Colab (notebooks/02_rl_colab.ipynb).
Given in full; what you write is rl/features.py and rl/reward.py (and the settings below).

    python rl/train_ppo.py --student T03                           # 8 parallel simulators, 3000 folds (~1 h on 8 cores)
    python rl/train_ppo.py --student T03 --n_envs 2 --steps 1000   # Colab free (2 CPUs): ~1.5 h
    python rl/train_ppo.py --student T03 --action xy --steps 20000 # the harder action space (extension)
    python rl/train_ppo.py --student T03 --resume runs/m3/ppo/ckpt.zip --steps 2000

One "step" here is one fold (~5 s of simulated robot motion, 0.5-1 s of CPU); an episode is 2 folds. Training starts
come from your team's seeds (scene.student_seeds); evaluation (rl/eval_policy.py) uses the shared held-out seeds.

Writes OUT/ckpt.zip (the final policy), OUT/ckpt_<steps>.zip (every --save_every), OUT/train_log.csv (one row per
episode: return, success, gaps), OUT/curve.png, OUT/config.json.
"""
import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sim')); sys.path.insert(0, str(ROOT / 'rl'))


def make(seeds, mesh, action, rank):
    def f():
        sys.path.insert(0, str(ROOT / 'sim')); sys.path.insert(0, str(ROOT / 'rl'))
        from fold_env import FoldEnv
        env = FoldEnv(seeds=seeds, mesh=mesh, action=action); env.reset(seed=rank); return env
    return f


def plot(log, path):
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        rows = list(csv.DictReader(open(log)))
        if not rows: return
        R = np.array([float(r['return']) for r in rows]); S = np.array([float(r['success']) for r in rows]); w = max(1, len(R) // 20)
        sm = lambda x: np.convolve(x, np.ones(w) / w, mode='valid')
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].plot(R, '.', ms=2, alpha=.3); ax[0].plot(np.arange(len(sm(R))) + w - 1, sm(R)); ax[0].set(xlabel='episode', ylabel='return')
        ax[1].plot(np.arange(len(sm(S))) + w - 1, sm(S) * 100); ax[1].set(xlabel='episode', ylabel=f'success % (moving avg {w})', ylim=(0, 100))
        fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    except Exception as e: print('plot failed:', e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--student', required=True); ap.add_argument('--mesh', default=None, help='cloth mesh folder (default: assets/cloth_example)')
    ap.add_argument('--action', choices=['keypoint', 'xy'], default='keypoint')
    ap.add_argument('--steps', type=int, default=3000, help='folds to train on (2 per episode)')
    ap.add_argument('--n_envs', type=int, default=8, help='parallel simulators (CPU processes)')
    ap.add_argument('--n_steps', type=int, default=16, help='PPO: folds per simulator between updates')
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--ent', type=float, default=0.01)
    ap.add_argument('--epochs', type=int, default=10); ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--save_every', type=int, default=500); ap.add_argument('--resume', type=Path)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', type=Path, default=ROOT / 'runs' / 'm3' / 'ppo')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from scene import student_seeds
    torch.set_num_threads(1)
    seeds = list(student_seeds(a.student))
    fns = [make(seeds, a.mesh, a.action, a.seed * 100 + i) for i in range(a.n_envs)]
    venv = SubprocVecEnv(fns) if a.n_envs > 1 else DummyVecEnv(fns)
    cfg = dict(vars(a), out=str(a.out), resume=str(a.resume) if a.resume else None, seeds=[seeds[0], seeds[-1]], obs_dim=venv.observation_space.shape[0],
               started=time.strftime('%Y-%m-%d %H:%M:%S'))
    (a.out / 'config.json').write_text(json.dumps(cfg, indent=2))
    if a.resume: model = PPO.load(a.resume, env=venv, device='cpu')
    else:
        model = PPO('MlpPolicy', venv, n_steps=a.n_steps, batch_size=a.batch, n_epochs=a.epochs, learning_rate=a.lr, ent_coef=a.ent,
                    gamma=0.99, gae_lambda=0.95, clip_range=0.2, policy_kwargs=dict(net_arch=[128, 128]), seed=a.seed, device='cpu', verbose=0)
    log = a.out / 'train_log.csv'; new = not (a.resume and log.exists())
    f = open(log, 'w' if new else 'a', newline=''); w = csv.writer(f)
    if new: w.writerow(['episode', 'folds', 'time_s', 'return', 'success', 'gap_a', 'gap_b', 'cover', 'env_seed'])
    t0 = time.time(); ret = np.zeros(a.n_envs); n_ep = [0]

    class Log(BaseCallback):
        def _on_step(self):
            for i, (r, d, info) in enumerate(zip(self.locals['rewards'], self.locals['dones'], self.locals['infos'])):
                ret[i] += r
                if d:
                    n_ep[0] += 1
                    w.writerow([n_ep[0], self.num_timesteps, round(time.time() - t0), round(ret[i], 4), int(info['success']), round(info['gap_a'], 1),
                                round(info['gap_b'], 1), round(info['cover'], 3), info['env_seed']]); ret[i] = 0
            if n_ep[0] and self.n_calls % 4 == 0:
                f.flush(); rows = list(csv.DictReader(open(log)))[-50:]
                print(f'\r  {self.num_timesteps:6d} folds  {len(rows) and n_ep[0]:5d} episodes  {time.time() - t0:6.0f} s   last 50: return '
                      f'{np.mean([float(r["return"]) for r in rows]):5.2f}  success {np.mean([float(r["success"]) for r in rows]) * 100:5.1f} %', end='', flush=True)
            if self.num_timesteps // a.save_every != (self.num_timesteps - a.n_envs) // a.save_every:
                self.model.save(a.out / f'ckpt_{self.num_timesteps}.zip'); f.flush(); plot(log, a.out / 'curve.png')
            return True

    print(f'PPO, {a.n_envs} simulators, {a.steps} folds, action {a.action}, obs {cfg["obs_dim"]} dims -> {a.out}')
    model.learn(total_timesteps=a.steps, callback=Log(), reset_num_timesteps=not a.resume)
    model.save(a.out / 'ckpt.zip'); f.close(); plot(log, a.out / 'curve.png'); venv.close()
    print(f'\ndone in {time.time() - t0:.0f} s -> {a.out / "ckpt.zip"}, {log}, {a.out / "curve.png"}')


if __name__ == '__main__':
    main()
