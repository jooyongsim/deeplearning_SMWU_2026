"""One episode file (.npz) for every stage: teleop.py, the evaluation and the expert write it; replay / video read it.

Keys (T = control steps at 10 Hz; step k is the state *before* action k):
    obs      (T, 80)    float32  env observation (scene.OBS_LAYOUT)
    act      (T, 4)     float32  executed action [vx, vy, vz, grip] in [-1, 1]
    human    (T,)       bool     a key was held at this step (teleop; False for scripted / policy steps)
    kp       (T, 25, 3) float32  cloth keypoints xyz (table frame, m)
    tcp      (T, 3)     float32  gripper TCP xyz
    gaps     (T, 2)     float32  after action k: corner gaps c0-c2, c1-c3 (mm)
    cover    (T,)       float32  after action k: covered area / flat area
    succ     (T,)       bool     after action k: success (sim/fold_metrics.py)
    qpos     (T, nq)    float32  full MuJoCo state (render_video.py draws it without simulating)
    picks    (P, 4)     float32  pick-and-place actions (pick xy, place xy, table frame) if the episode was made of them
    seed, mesh, length, success, student, source, version
"""
import time

import numpy as np

VERSION = 1
KEYS = ('obs', 'act', 'human', 'kp', 'tcp', 'gaps', 'cover', 'succ', 'qpos', 'picks', 'seed', 'mesh', 'length', 'success', 'student', 'source', 'version')


class Recorder:
    def __init__(self, env, seed, mesh, student, source):
        self.env, self.meta = env, dict(seed=seed, mesh=str(mesh), student=str(student), source=source)
        self.rows = {k: [] for k in ('obs', 'act', 'human', 'kp', 'tcp', 'gaps', 'cover', 'succ', 'qpos')}; self.picks = []

    def before(self, obs, human=False):
        e, r = self.env, self.rows
        r['obs'].append(np.asarray(obs, np.float32)); r['kp'].append(e.keypoints().astype(np.float32)); r['tcp'].append(e.tcp().astype(np.float32))
        r['qpos'].append(e.d.qpos.astype(np.float32)); r['human'].append(bool(human))

    def after(self, info):
        r = self.rows; r['act'].append(np.asarray(self.env.last_a, np.float32))
        r['gaps'].append(np.array([info['gap_a'], info['gap_b']], np.float32)); r['cover'].append(np.float32(info['cover'])); r['succ'].append(bool(info['success']))

    def callback(self):
        """For primitives.pick_place(..., cb=rec.callback())."""
        def cb(stage, x):
            if stage == 'before': self.before(self.env.obs())
            else: self.after(x)
        return cb

    def __len__(self): return len(self.rows['act'])

    def save(self, path):
        n = len(self); r = {k: np.array(v[:n]) for k, v in self.rows.items()}
        np.savez_compressed(path, **r, picks=np.array(self.picks, np.float32).reshape(-1, 4), length=np.int64(n),
                            success=bool(r['succ'][-1]) if n else False, seed=np.int64(self.meta['seed']), mesh=self.meta['mesh'],
                            student=self.meta['student'], source=self.meta['source'], version=np.int64(VERSION), created=time.strftime('%Y-%m-%d %H:%M:%S'))
        return path


def load(path):
    z = np.load(path, allow_pickle=False)
    return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


def problems(ep):
    out = [f'missing key {k}' for k in KEYS if k not in ep]
    if out: return out
    T = int(ep['length'])
    shapes = dict(obs=(T, 80), act=(T, 4), human=(T,), kp=(T, 25, 3), tcp=(T, 3), gaps=(T, 2), cover=(T,), succ=(T,))
    out += [f'{k}: shape {ep[k].shape}, expected {s}' for k, s in shapes.items() if ep[k].shape != s]
    if out: return out
    if T < 5: out.append(f'only {T} steps')
    if not np.isfinite(ep['obs']).all(): out.append('obs not finite')
    if np.abs(ep['act']).max() > 1 + 1e-5: out.append('act outside [-1, 1]')
    return out
