"""Build the robot + cloth scene and reset it to a seeded start. Every script (check_env, teleop, eval, RL) builds its
episode here, so the same seed and mesh give the same start everywhere.

    from scene import make_env
    env, obs = make_env(seed=1234)                           # the example cloth (assets/cloth_example)
    env, obs = make_env(seed=1234, mesh='assets/my_cloth')   # your own cloth (Milestone 1)
    obs, reward, done, info = env.step([vx, vy, vz, grip])   # action in [-1, 1]^4, 10 Hz

The start: the cloth flat on the table, turned by up to 30 deg and moved by up to 3 cm (drawn from the seed); the
gripper open, 15 cm above the cloth's centre.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MESH = ROOT / 'assets' / 'cloth_example'
OBS_DIM, ACT_DIM = 80, 4
# the observation, positions in the table frame (m) relative to (cloth_env.CENTRE, 0)
OBS_LAYOUT = {'cloth_kp_xyz': slice(0, 75),     # 25 cloth keypoints x 3: a 5 x 5 grid on the cloth, row by row from corner c0
                                                # (kp 0 = c0, 4 = c1, 24 = c2, 20 = c3, 12 = centre)
              'tcp_xyz': slice(75, 78),         # gripper TCP
              'grip_closed': slice(78, 79),     # 1 = closed
              'held': slice(79, 80)}            # vertices held / GRASP_K (0 = holding nothing)
CORNER_KP = [0, 4, 24, 20]                      # keypoint index of corners c0, c1, c2, c3

# seeds as Assignment 01: each team's teleop / training starts from its own block, the evaluation from a shared block
SEEDS_PER_STUDENT = 100
EVAL_SEEDS = range(900_000, 900_050)

_ENVS = {}


def student_seeds(student):
    """T03 -> 300-399."""
    digits = ''.join(ch for ch in str(student) if ch.isdigit())
    if not digits: raise ValueError(f'student id {student!r} has no digits')
    base = int(digits[-4:]) * SEEDS_PER_STUDENT; return range(base, base + SEEDS_PER_STUDENT)


def make_env(seed, mesh=None, horizon=1200):
    """(env, first observation) for this seed. The model is compiled once per mesh and reused (a new seed only moves
    the cloth), so calling make_env again is cheap."""
    from cloth_env import ClothFoldEnv
    key = str(Path(mesh or DEFAULT_MESH).resolve())
    if key not in _ENVS: _ENVS[key] = ClothFoldEnv(key)
    env = _ENVS[key]; env.horizon = horizon
    return env, env.reset(seed)
