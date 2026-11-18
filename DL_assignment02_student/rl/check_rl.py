#!/usr/bin/env python3
"""Milestone 3: check your rl/features.py and rl/reward.py before training (no training here; ~30 s).

    python rl/check_rl.py
Writes runs/m3/check_rl.json. Each test names what it checks; all must PASS.
"""
import json, sys, time, traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sim')); sys.path.insert(0, str(ROOT / 'rl'))


def info(ga, gb, cover, success=False, bad=False):
    return dict(gap_a=float(ga), gap_b=float(gb), cover=float(cover), folds=int(ga <= 70) + int(gb <= 70), success=success, held=0, bad=bad)


def tests(sim=True):
    import features, reward
    from scene import make_env
    env, obs = make_env(900_000); i0 = env.info
    out = {}

    def t(name):
        def deco(f):
            try: f(); out[name] = 'PASS'
            except NotImplementedError as e: out[name] = f'FAIL: not implemented ({e})'
            except Exception as e: out[name] = f'FAIL: {e!r}'
            return f
        return deco

    @t('features: 1-D, finite, float, the same length for k = 0, 1, 2')
    def _():
        f = [np.asarray(features.obs_features(obs, k, i0)) for k in range(3)]
        assert all(x.ndim == 1 for x in f), f'shapes {[x.shape for x in f]}'
        assert len({len(x) for x in f}) == 1, f'lengths {[len(x) for x in f]}'
        assert all(np.isfinite(x).all() for x in f) and np.issubdtype(f[0].dtype, np.floating)

    @t('features: roughly unit scale (|x| <= 10)')
    def _():
        x = np.asarray(features.obs_features(obs, 0, i0)); assert np.abs(x).max() <= 10, f'max |x| = {np.abs(x).max():.1f}: scale it'

    @t('features: change when the cloth moves')
    def _():
        o2 = obs.copy(); o2[0:75] += np.tile([.05, -.03, .004], 25)
        assert not np.allclose(features.obs_features(obs, 0, i0), features.obs_features(o2, 0, i0)), 'features ignore the cloth keypoints'

    @t('features: tell the policy which fold comes next (k = 0 and k = 1 differ for the same cloth)')
    def _():
        assert not np.allclose(features.obs_features(obs, 0, i0), features.obs_features(obs, 1, i0))

    i_start = info(424, 424, 1.0)

    @t('reward: a fold closing one corner pair (424 -> 20 mm) earns more than one that changes nothing')
    def _():
        good = reward.compute_reward(i_start, info(20, 424, .5), 1, False, i_start); none = reward.compute_reward(i_start, i_start, 1, False, i_start)
        assert good > none + .1, f'closing a pair {good:.3f}, nothing {none:.3f}'

    @t('reward: doing nothing earns about 0 (|r| < 0.05)')
    def _():
        r = reward.compute_reward(i_start, i_start, 1, False, i_start); assert abs(r) < .05, f'r = {r:.3f}'

    @t('reward: making it worse earns less than doing nothing')
    def _():
        mid = info(200, 424, .6)
        assert reward.compute_reward(mid, info(350, 424, .9), 1, False, i_start) < reward.compute_reward(mid, mid, 1, False, i_start)

    @t('reward: finishing successfully earns more than finishing at the same gaps without success')
    def _():
        p = info(20, 424, .5)
        assert reward.compute_reward(p, info(40, 50, .2, True), 2, True, i_start) > reward.compute_reward(p, info(40, 50, .2, False), 2, True, i_start)

    @t('reward: a broken simulation is punished')
    def _():
        assert reward.compute_reward(i_start, info(424, 424, 1., bad=True), 1, True, i_start) < -.5

    @t('reward: finite floats')
    def _():
        r = reward.compute_reward(i_start, info(20, 424, .5), 1, False, i_start); assert np.isfinite(float(r))

    if sim:
        @t('FoldEnv: one episode of 2 random folds runs (spaces, finite observations and rewards)')
        def _():
            from fold_env import FoldEnv
            e = FoldEnv(seeds=[900_001]); o, _ = e.reset(seed=0); assert e.observation_space.contains(o)
            for k in range(2):
                o, r, term, trunc, inf = e.step(e.action_space.sample()); assert np.isfinite(o).all() and np.isfinite(r)
            assert term, 'the episode should end after 2 folds'
    return out


def main():
    out = ROOT / 'runs' / 'm3'; out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    try: res = tests()
    except Exception: traceback.print_exc(); res = {'import': 'FAIL: rl/features.py or rl/reward.py does not import'}
    for k, v in res.items(): print(f'  [{"PASS" if v == "PASS" else "FAIL"}] {k}' + ('' if v == 'PASS' else f'\n         {v[6:]}'))
    ok = all(v == 'PASS' for v in res.values())
    (out / 'check_rl.json').write_text(json.dumps(dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), passed=ok, tests=res, wall_s=round(time.time() - t0, 1)), indent=2))
    print(f'{"PASSED" if ok else "FAILED"} ({sum(v == "PASS" for v in res.values())}/{len(res)}) -> {out / "check_rl.json"}'); sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
