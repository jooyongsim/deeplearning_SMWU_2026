"""The folding task as a Gymnasium environment for reinforcement learning (given in full).

One RL step = one pick-and-place fold executed by the robot (sim/primitives.py: ~5 s of 10 Hz robot control). An
episode has N_FOLDS = 2 steps. Two action spaces:

    --action keypoint  (default)  MultiDiscrete([25, 25]): pick at cloth keypoint i, place onto cloth keypoint j
                                  (keypoints: the 5 x 5 grid of scene.OBS_LAYOUT; 0 = c0, 4 = c1, 24 = c2, 20 = c3)
    --action xy                   Box(-1, 1, (4,)): pick xy, place xy anywhere on the workspace,
                                  CENTRE + XY_RANGE x action  (harder: the policy must also find the cloth)

The observation and the reward are YOURS (Milestone 3): rl/features.py and rl/reward.py.

    env = FoldEnv(seeds=range(300, 400))         # each reset draws a seed from these (your team's block)
    obs, info = env.reset(); obs, r, terminated, truncated, info = env.step(env.action_space.sample())
"""
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sim')); sys.path.insert(0, str(ROOT / 'rl'))

N_FOLDS, XY_RANGE = 2, 0.25


class FoldEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, seeds=None, mesh=None, action='keypoint', record=None):
        """seeds: the seeds resets draw from (None: any); record: a function(env) -> Recorder, to save the episode."""
        from scene import make_env
        import features
        self.make_env, self.mesh, self.mode, self.record = make_env, mesh, action, record
        self.seeds = None if seeds is None else list(seeds)
        self.env, obs = make_env(0 if self.seeds is None else self.seeds[0], mesh)
        self.action_space = gym.spaces.MultiDiscrete([25, 25]) if action == 'keypoint' else gym.spaces.Box(-1, 1, (4,), np.float32)
        dim = len(features.obs_features(obs, 0, self.env.info))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (dim,), np.float32)
        self.rec = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        import features
        s = (options or {}).get('env_seed')
        if s is None: s = int(self.np_random.choice(self.seeds)) if self.seeds is not None else int(self.np_random.integers(0, 2 ** 31 - 1))
        self.env_seed = s; self.env, obs = self.make_env(s, self.mesh); self.k = 0
        self.rec = self.record(self.env, s) if self.record else None
        self.info0 = dict(self.env.info); self.prev = dict(self.env.info); self.last_obs = obs
        return features.obs_features(obs, 0, self.env.info).astype(np.float32), dict(self.env.info, env_seed=s)

    def points(self, action):
        """The action -> (pick xy, place xy) in the table frame."""
        from cloth_env import CENTRE
        if self.mode == 'keypoint':
            kp = self.env.keypoints()[:, :2]; i, j = int(action[0]), int(action[1]); return kp[i], kp[j]
        a = np.clip(np.asarray(action, float), -1, 1); return CENTRE + XY_RANGE * a[:2], CENTRE + XY_RANGE * a[2:]

    def step(self, action):
        import features, reward
        from primitives import pick_place
        pick, place = self.points(action)
        if self.rec is not None: self.rec.picks.append(np.r_[pick, place])
        info = pick_place(self.env, pick, place, self.rec.callback() if self.rec is not None else None)
        self.k += 1
        terminated = self.k >= N_FOLDS or info['bad']
        r = float(reward.compute_reward(self.prev, info, self.k, terminated, self.info0))
        self.prev = dict(info); obs = self.env.obs(); self.last_obs = obs
        return features.obs_features(obs, self.k, info).astype(np.float32), r, terminated, False, dict(info, pick=pick, place=place, env_seed=self.env_seed)
