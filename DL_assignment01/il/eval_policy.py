#!/usr/bin/env python3
"""Milestone 4: run a policy closed loop in the simulator on the shared held-out starts (scene.EVAL_SEEDS, no demo uses
them) and score it. Given in full; you only provide the checkpoint (policy.load_agent).

    python il/eval_policy.py --agent zero                          # baseline: stands still (checks the harness, ~0 %)
    python il/eval_policy.py --agent random --n 5                  # baseline: random actions
    python il/eval_policy.py --agent bc --ckpt runs/m3/ckpt.pt     # your policy, all 50 starts
    python il/eval_policy.py --agent bc --ckpt runs/m3/ckpt.pt --n 3 --view    # watch it

Success = sim/success.py's (position <= 50 mm, rotation <= 15 deg, shape <= 12 mm) at the episode's last step; the
report also has 'ever' (at any step) and the final errors.
Writes OUT/eval.json and OUT/rollouts/ep_<seed>.npz (episode_io format, source 'eval:<agent>').
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sim'))


class ZeroAgent:
    def reset(self): pass
    def act(self, obs): return np.zeros(3)


class RandomAgent:
    def __init__(self, seed=0): self.rng = np.random.default_rng(seed)
    def reset(self): pass
    def act(self, obs): return self.rng.uniform(-1, 1, 3)


def rollout(agent, seed, steps, out, name, view=False):
    from scene import make_env
    from episode_io import Recorder
    env, obs = make_env(seed, horizon=steps); rec = Recorder(env, seed, env.p['mu_table'], name, f'eval:{name}')
    agent.reset(); v = None; info = {}
    if view:
        import mujoco.viewer
        from rope_env import CENTRE
        v = mujoco.viewer.launch_passive(env.m, env.d); v.cam.lookat[:], v.cam.distance, v.cam.azimuth, v.cam.elevation = np.r_[CENTRE, 0], .9, 90., -55.
    t0 = time.perf_counter()
    for k in range(steps):
        a = np.clip(np.asarray(agent.act(obs), float).reshape(3), -1, 1)
        rec.before(obs); obs, _, done, info = env.step(a); rec.after(info, True)
        if v is not None:
            if not v.is_running(): break
            v.sync(); time.sleep(max(.1 - (time.perf_counter() - t0), 0)); t0 = time.perf_counter()
        if done: break
    if v is not None: v.close()
    rec.save(out / f'ep_{seed}.npz')
    e = info['err']; return dict(seed=seed, steps=len(rec), success=bool(info['success']), ever=bool(any(rec.rows['succ'])),
                                 pos_mm=round(e['pos'], 1), rot_deg=round(e['rot'], 1), shape_mm=round(e['shape'], 1), blocked_end=bool(info['blocked']))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--agent', choices=['zero', 'random', 'bc'], required=True); ap.add_argument('--ckpt', type=Path)
    ap.add_argument('--n', type=int, help='first n of the 50 eval seeds (default all)'); ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--out', type=Path, help='default runs/m4/<agent>'); ap.add_argument('--view', action='store_true')
    a = ap.parse_args()
    from scene import EVAL_SEEDS
    if a.agent == 'bc':
        if a.ckpt is None: ap.error('--agent bc needs --ckpt')
        sys.path.insert(0, str(ROOT / 'il')); from policy import load_agent
        agent = load_agent(a.ckpt)
    else: agent = ZeroAgent() if a.agent == 'zero' else RandomAgent()
    out = a.out or ROOT / 'runs' / 'm4' / a.agent; (out / 'rollouts').mkdir(parents=True, exist_ok=True)
    seeds = list(EVAL_SEEDS)[:a.n] if a.n else list(EVAL_SEEDS); res = []
    for s in seeds:
        r = rollout(agent, s, a.steps, out / 'rollouts', a.agent, a.view); res.append(r)
        print(f'seed {s}: {"SUCCESS" if r["success"] else "fail   "}  {r["steps"]:4d} steps  pos {r["pos_mm"]:6.1f} mm  rot {r["rot_deg"]:5.1f} deg  shape {r["shape_mm"]:5.1f} mm', flush=True)
    sr = float(np.mean([r['success'] for r in res])); ever = float(np.mean([r['ever'] for r in res]))
    rep = dict(agent=a.agent, ckpt=str(a.ckpt) if a.ckpt else None, n=len(res), success_rate=sr, ever_success_rate=ever,
               median_final=dict(pos_mm=float(np.median([r['pos_mm'] for r in res])), rot_deg=float(np.median([r['rot_deg'] for r in res])),
                                 shape_mm=float(np.median([r['shape_mm'] for r in res]))), seeds=seeds, episodes=res, time=time.strftime('%Y-%m-%d %H:%M:%S'))
    (out / 'eval.json').write_text(json.dumps(rep, indent=2))
    print(f'\n{a.agent}: success {sr * 100:.0f} % ({sum(r["success"] for r in res)}/{len(res)}), ever {ever * 100:.0f} % -> {out / "eval.json"}')


if __name__ == '__main__':
    main()
