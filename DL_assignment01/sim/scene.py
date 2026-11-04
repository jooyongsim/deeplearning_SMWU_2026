"""Build the robot + rope-T scene and reset it to a seeded start. Every script (check_env, teleop, replay, eval) builds
its episode here, so the same seed and friction give the same start everywhere.

    from scene import make_env
    env, obs = make_env(seed=1234)          # obs: (85,) float32, see OBS_LAYOUT
    obs, reward, done, info = env.step([vx, vy, vz])     # action in [-1, 1]^3, 10 Hz

The start: the reference demo's rope moved by up to 3 cm and turned by up to 30 deg (drawn from the seed); the robot at
that demo's start pose. Table friction: the fitted value (data/fit_t.json) unless mu is given.
"""
import json

import numpy as np

import rope_env as RE
from demo_data import load, robot_setup

REF = '20261008-220352'    # the reference demo whose start pose the seeded starts perturb
OBS_DIM, ACT_DIM = 85, 3
# the observation, all positions in the table frame (m) relative to rope_env.CENTRE
OBS_LAYOUT = {'rope_kp_xy': slice(0, 38),       # 19 rope keypoints x 2: arm 0-4, arm 5-9, stem 10-18 (0, 5, 10 = junction)
              'goal_kp_xy': slice(38, 76),      # the goal T's 19 keypoints, same order (fixed in every episode)
              'tip_xyz': slice(76, 79),         # stick tip
              'joints': slice(79, 85)}          # UR5e joint angles / pi

# seeds: each student's demonstrations come from their own block (from the student id), the evaluation from a shared
# held-out block no demonstration uses, so every policy is scored on the same starts
SEEDS_PER_STUDENT = 100
EVAL_SEEDS = range(900_000, 900_050)


def student_seeds(student):
    """The block of seeds a student's demonstrations use: 100 x (the id's last 4 digits) onwards."""
    digits = ''.join(ch for ch in str(student) if ch.isdigit())
    if not digits: raise ValueError(f'student id {student!r} has no digits')
    base = int(digits[-4:]) * SEEDS_PER_STUDENT; return range(base, base + SEEDS_PER_STUDENT)


_CACHE = {}


def _setup():
    if not _CACHE:
        D, names, shapes, goal = load(); robot = robot_setup(D, names)
        _CACHE.update(names=names, shapes=shapes, goal=goal, robot=robot)
    return _CACHE


def make_env(seed, mu=None, horizon=600):
    """(env, first observation) for this seed. mu: table friction (default: the fitted one)."""
    from robot_env import RobotRopeEnv
    c = _setup(); robot, ref = c['robot'], c['names'].index(REF)
    env = RobotRopeEnv(c['shapes'], c['goal'], robot['starts'], sticks=robot['sticks']); env.horizon = horizon
    fit = json.loads(RE.FIT.read_text())
    if mu is not None: fit['mu_table'] = fit['friction'] = float(mu)
    env.p = fit
    env.rng = np.random.default_rng(seed); env.perturb, env.scale, env.stick_jitter = True, 1., False
    env.rot_max, env.off_max, env.off_disc = np.radians(30), .03, True; env.stick = robot['sticks'][ref]
    return env, env.reset(index=ref)
