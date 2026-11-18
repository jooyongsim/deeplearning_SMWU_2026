"""Pick-and-place: one fold as a sequence of 10 Hz robot actions (env.step), so a fold made this way is exactly what a
person could do with the keyboard. The RL environment (rl/fold_env.py), the evaluation and the videos all use it.

    from primitives import pick_place
    info = pick_place(env, pick_xy, place_xy)        # table-frame metres; returns the last env.step info

The motion: open, move above the pick point, go down to the top of the cloth there, close (grasp), carry along an
arc whose height grows with the distance (lifting the cloth so it lays over instead of being dragged), lower at the
place point, open, go back up, and wait SETTLE steps for the cloth to come to rest.
"""
import numpy as np

from cloth_env import VMAX, CTRL_DT, WORK

HOVER, PLACE_Z, SETTLE = 0.08, 0.012, 8
ARC = (0.28, 0.04, 0.14)          # arc height = clip(0.28 x distance, 4 cm, 14 cm)


def _go(env, target, grip, speed, cb, max_steps=60, tol=.002):
    """Straight line to target at speed (fraction of VMAX); returns the last info."""
    info = env.info
    for _ in range(max_steps):
        d = np.asarray(target) - env.tcp()
        if np.linalg.norm(d) < tol: break
        v = d / (VMAX * CTRL_DT); n = np.abs(v).max()
        if n > speed: v *= speed / n
        info = _step(env, np.r_[v, grip], cb)
        if info['bad']: break
    return info


def _step(env, a, cb):
    if cb: cb('before', a)
    obs, r, done, info = env.step(a)
    if cb: cb('after', info)
    return info


def pick_place(env, pick, place, cb=None, speed=1., arc=None):
    """cb(stage, x): called around every env.step ('before', action) / ('after', info), e.g. to record the episode."""
    pick, place = np.asarray(pick, float)[:2], np.asarray(place, float)[:2]
    pick, place = np.clip(pick, WORK[0, :2], WORK[1, :2]), np.clip(place, WORK[0, :2], WORK[1, :2])
    _go(env, np.r_[pick, HOVER], -1, speed, cb)
    X = env.verts(); near = np.linalg.norm(X[:, :2] - pick, axis=1) < .03
    z = max(float(X[near, 2].max()) + .002 if near.any() else .003, .003)        # the top of the cloth (layers) there
    _go(env, np.r_[pick, z], -1, speed, cb)
    _step(env, np.r_[0, 0, 0, 1], cb)                                              # close: grasp
    p0 = env.tcp(); D = np.linalg.norm(place - p0[:2]); A = arc or ARC; h = float(np.clip(A[0] * D, A[1], A[2]))
    p1 = np.r_[place, PLACE_Z]; L = np.pi / 2 * max(D, 2 * h); n = max(int(np.ceil(L / (VMAX * CTRL_DT * speed))), 3)
    for i in range(1, n + 1):                                                      # arc: one waypoint per step
        u = i / n; w = p0 + (p1 - p0) * u; w[2] += h * np.sin(np.pi * u)
        v = (w - env.tcp()) / (VMAX * CTRL_DT); _step(env, np.r_[np.clip(v, -1, 1), 1], cb)
    _go(env, p1, 1, speed, cb, max_steps=10)
    _step(env, np.r_[0, 0, 0, -1], cb)                                             # open: release
    _go(env, np.r_[place, HOVER], -1, speed, cb, max_steps=10)
    info = env.info
    for _ in range(SETTLE): info = _step(env, np.r_[0, 0, 0, -1], cb)
    return info
