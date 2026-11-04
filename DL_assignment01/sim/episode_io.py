"""One episode file (.npz) for every stage: teleop.py writes it, replay_demo.py / inspect_demos.py read it, the imitation
dataset is built from it, eval_policy.py writes the same format for its rollouts.

Keys (T = steps; step k is the state *before* action k):
    obs      (T, 85)    float32  env observation (scene.OBS_LAYOUT): rope kp xy, goal kp xy, stick tip xyz, joints / pi
    act      (T, 3)     float32  executed action, normalised stick-tip velocity in [-1, 1] (base frame, 10 Hz)
    human    (T,)       bool     a key was held at this step (False: idle steps you may want to drop)
    kp       (T, 19, 2) float32  rope keypoints xy (table frame, m)
    kp_z     (T, 19)    float32  their heights
    tip      (T, 3)     float32  stick tip xyz
    err      (T, 3)     float32  after action k: position (mm), rotation (deg), shape (mm) error to the goal T
    succ     (T,)       bool     after action k: success (sim/success.py thresholds)
    goal     (19, 2)    float32  goal keypoints xy
    qpos     (T, nq)    float64  full MuJoCo state (replay_demo.py --view shows it without simulating)
    seed, mu, length, success, student, source, version
"""
import time

import numpy as np

VERSION = 1
KEYS = ('obs', 'act', 'human', 'kp', 'kp_z', 'tip', 'err', 'succ', 'goal', 'qpos', 'seed', 'mu', 'length', 'success', 'student', 'source', 'version')


class Recorder:
    def __init__(self, env, seed, mu, student, source):
        self.env, self.meta = env, dict(seed=seed, mu=mu, student=str(student), source=source)
        self.rows = {k: [] for k in ('obs', 'act', 'human', 'kp', 'kp_z', 'tip', 'err', 'succ', 'qpos')}

    def before(self, obs):
        """Call with the observation, before env.step."""
        e, r = self.env, self.rows
        r['obs'].append(np.asarray(obs, np.float32)); r['kp'].append(np.asarray(e.kp, np.float32)); r['kp_z'].append(np.asarray(e.kp_z, np.float32))
        r['tip'].append(np.asarray(e.tip_of(e.q), np.float32)); r['qpos'].append(e.d.qpos.copy())

    def after(self, info, human=True):
        """Call with env.step's info, after it."""
        r, x = self.rows, info['err']
        r['act'].append(np.asarray(self.env.last_a, np.float32)); r['human'].append(bool(human))
        r['err'].append(np.array([x['pos'], x['rot'], x['shape']], np.float32)); r['succ'].append(bool(info['success']))

    def __len__(self): return len(self.rows['act'])

    def save(self, path):
        n = len(self); r = {k: np.array(v[:n]) for k, v in self.rows.items()}
        np.savez_compressed(path, **r, goal=np.asarray(self.env.goal, np.float32), length=np.int64(n),
                            success=bool(r['succ'][-1]) if n else False, seed=np.int64(self.meta['seed']), mu=np.float64(self.meta['mu']),
                            student=self.meta['student'], source=self.meta['source'], version=np.int64(VERSION), created=time.strftime('%Y-%m-%d %H:%M:%S'))
        return path


def load(path):
    """The episode as a dict of arrays (scalars as Python values)."""
    z = np.load(path, allow_pickle=False)
    return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


def problems(ep):
    """What is wrong with an episode dict (empty list: valid)."""
    out = [f'missing key {k}' for k in KEYS if k not in ep]
    if out: return out
    T = int(ep['length'])
    shapes = dict(obs=(T, 85), act=(T, 3), human=(T,), kp=(T, 19, 2), kp_z=(T, 19), tip=(T, 3), err=(T, 3), succ=(T,), goal=(19, 2))
    out += [f'{k}: shape {ep[k].shape}, expected {s}' for k, s in shapes.items() if ep[k].shape != s]
    if out: return out
    if T < 10: out.append(f'only {T} steps')
    if not np.isfinite(ep['obs']).all(): out.append('obs not finite')
    if np.abs(ep['act']).max() > 1 + 1e-5: out.append('act outside [-1, 1]')
    if not np.allclose(ep['obs'][:, 0:38].reshape(T, 19, 2) + [-.15, -.45], ep['kp'], atol=1e-4): out.append('obs rope keypoints do not match kp')
    return out
