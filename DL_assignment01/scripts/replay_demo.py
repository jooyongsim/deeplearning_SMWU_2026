#!/usr/bin/env python3
"""Milestone 2: check a saved episode. Re-simulates it from its seed with its recorded actions (open loop, no window)
and compares the rope keypoints with the recorded ones; or plays its recorded states in the viewer.

    python scripts/replay_demo.py demos/20261234/ep_123400.npz            # re-simulate, print the drift
    python scripts/replay_demo.py demos/20261234/ep_123400.npz --view     # watch it (recorded qpos, 10 Hz)

The same computer and libraries reproduce an episode exactly (drift 0.0 mm); another CPU or BLAS can differ by 1e-7 at a
step, which contact-rich pushing can grow to centimetres. So a large drift on another machine is not by itself an error.
"""
import argparse, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sim'))


def resim(ep):
    """Recorded actions re-run from the episode's seed: (keypoints (T, 19, 2), success at the end)."""
    from scene import make_env
    env, _ = make_env(int(ep['seed']), float(ep['mu']), int(ep['length']) + 1); KP, info = [], {}
    for a in ep['act']:
        KP.append(env.kp.copy()); _, _, done, info = env.step(a)
        if done: break
    return np.array(KP), bool(info.get('success', False))


def view(ep):
    import mujoco, mujoco.viewer
    from scene import make_env
    env, _ = make_env(int(ep['seed']), float(ep['mu']))
    with mujoco.viewer.launch_passive(env.m, env.d) as v:
        for q in ep['qpos']:
            if not v.is_running(): break
            with v.lock(): env.d.qpos[:] = q; mujoco.mj_forward(env.m, env.d)
            v.sync(); time.sleep(.1)
        while v.is_running(): time.sleep(.1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('episode', type=Path); ap.add_argument('--view', action='store_true')
    a = ap.parse_args()
    from episode_io import load, problems
    ep = load(a.episode); bad = problems(ep)
    if bad: sys.exit(f'{a.episode}: ' + '; '.join(bad))
    print(f'{a.episode.name}: seed {ep["seed"]}, {ep["length"]} steps ({ep["human"].sum()} driven), success {ep["success"]}')
    if a.view: return view(ep)
    kp, ok = resim(ep); n = min(len(kp), len(ep['kp']))
    d = np.linalg.norm(kp[:n] - ep['kp'][:n], axis=2).mean(1) * 1000
    print(f'  re-simulated {n} steps: keypoint drift mean {d.mean():.2f} mm, max {d.max():.2f} mm, at the end {d[-1]:.2f} mm; success {ok}')


if __name__ == '__main__':
    main()
