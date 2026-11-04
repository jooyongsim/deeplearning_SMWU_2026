"""Rope-T success as three errors against the goal T (no MuJoCo; the simulator, the evaluations and the real-robot runner
share it):
  pos   distance between the keypoints' centroids (mm)
  rot   rotation of the best rigid fit of the keypoints onto the goal's (deg)
  shape keypoints' mean distance from the goal's after that rigid fit (mm): how far the rope is from a T, wherever it is
The two arms are matched either way round (as goal_distance).

THRESH (2026-10-15), from the real demos' final tracked shapes, where the operator stopped each one as done (8 of 10;
221609 and 221816 end with the T tracked turned 71 / 96 deg, a tracking failure): pos 5-19 mm, rot 1.5-3.9 deg, shape
1.2-3.3 mm; plus the cameras' noise on a still rope (real runs, 1.5 s): pos +-3 mm, rot +-2 deg, shape +-2 mm; and rl1
on the real robot, judged a success, over its last 10 s: pos 12, rot 4, shape 6.6 mm (median).
2026-10-18: loosened to 50 mm / 15 deg / 12 mm (the professor: a looser success in the sim and on the real robot).
Position at 30 mm (was 20): the operator stopped demos at up to 19 mm, the cameras add a few, and 20 made a rope
at 19 mm flicker between success and not.
"""
import os

import numpy as np

N0, N1, N = 5, 5, 19
SWAP = np.r_[np.arange(N0, N0 + N1), np.arange(N0), np.arange(N0 + N1, N)]
THRESH = dict(pos=50., rot=15., shape=12.)      # mm, deg, mm (2026-10-18, looser as asked; 30 / 8 / 8 and 20 / 8 / 8 before)
if os.environ.get('ROPE_SUCC_TH'):              # 2026-10-22: a stricter criterion for a study, e.g. ROPE_SUCC_TH=30,8,8
    THRESH = dict(zip(('pos', 'rot', 'shape'), map(float, os.environ['ROPE_SUCC_TH'].split(','))))


def errors(kp, goal):
    """{'pos': mm, 'rot': deg, 'shape': mm} of 19 xy keypoints against the goal's."""
    best = None
    for k in (kp, kp[SWAP]):
        a, b = k - k.mean(0), goal - goal.mean(0)
        U, _, Vt = np.linalg.svd(a.T @ b); R = (U @ Vt).T
        if np.linalg.det(R) < 0: U[:, -1] *= -1; R = (U @ Vt).T     # a rotation, no reflection
        res = np.linalg.norm(a @ R.T - b, axis=1).mean()
        e = dict(pos=float(np.linalg.norm(k.mean(0) - goal.mean(0)) * 1000), rot=float(abs(np.degrees(np.arctan2(R[1, 0], R[0, 0])))),
                 shape=float(res * 1000), mean=float(np.linalg.norm(k - goal, axis=1).mean() * 1000))
        if best is None or e['mean'] < best['mean']: best = e
    return best


def is_success(kp, goal, th=None):
    th = th or THRESH; e = errors(kp, goal)
    return all(e[k] <= th[k] for k in ('pos', 'rot', 'shape'))
