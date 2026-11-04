"""Milestone 3 (part 1): demonstrations -> (observation, action) training pairs.

Given (do not change):  load_demos, split_by_episode
TODO (you):             make_features, make_pairs, Normalizer, BCDataset

The policy sees positions only: rope keypoints, goal keypoints, stick tip (all in each episode's 'obs', see
sim/scene.py OBS_LAYOUT), and predicts the stick-tip velocity 'act' (3,) in [-1, 1].
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sim'))
from episode_io import load, problems          # noqa: E402
from scene import OBS_LAYOUT                    # noqa: E402


# ---------------------------------------------------------------- given
PER_STEP = ('obs', 'act', 'human', 'kp', 'kp_z', 'tip', 'err', 'succ', 'qpos')     # the keys with one row per step


def load_demos(folders, drop_idle=True, only_success=False):
    """Every valid ep_*.npz in the folders, as episode dicts (sim/episode_io.py keys). drop_idle: remove the steps where
    no key was held (the operator thinking is not an action to imitate). only_success: keep successful episodes only."""
    eps = []
    for d in folders:
        for f in sorted(Path(d).glob('ep_*.npz')):
            ep = load(f)
            if problems(ep) or (only_success and not ep['success']): continue
            if drop_idle:
                h = ep['human'].astype(bool)
                ep = {k: (v[h] if k in PER_STEP else v) for k, v in ep.items()}
                ep['length'] = int(h.sum())
            if ep['length'] > 0: ep['name'] = f.stem; eps.append(ep)
    return eps


def split_by_episode(eps, val_frac=.2, seed=0):
    """(train, val) lists of episodes. Split by episode, not by step: neighbouring steps are almost the same, so a
    step-wise split puts near-copies of the training data into validation."""
    i = np.random.default_rng(seed).permutation(len(eps)); n_val = max(1, int(round(len(eps) * val_frac))) if len(eps) > 1 else 0
    return [eps[j] for j in i[n_val:]], [eps[j] for j in i[:n_val]]


# ---------------------------------------------------------------- TODO
def make_features(obs):
    """obs (T, 85) -> features (T, D) the policy is given.

    TODO: choose and build the position features. Start from
        rope keypoints xy  obs[:, OBS_LAYOUT['rope_kp_xy']]  (38)
        goal keypoints xy  obs[:, OBS_LAYOUT['goal_kp_xy']]  (38)
        stick tip xyz      obs[:, OBS_LAYOUT['tip_xyz']]     (3)
    Ideas to try (explain your choice in the report): rope minus goal, the tip relative to the nearest keypoint, the
    rope's centroid and heading. Do not use anything eval_policy.py cannot give you at run time (it calls this on the
    env's observation)."""
    raise NotImplementedError('TODO: il/dataset.py make_features')


def make_pairs(eps, obs_horizon=1, act_horizon=1):
    """Episodes -> X (N, obs_horizon * D), Y (N, act_horizon * 3) float32.

    Sample t takes the features of steps t - obs_horizon + 1 .. t (repeat step 0 before the start) and the actions of
    steps t .. t + act_horizon - 1 (repeat the last action after the end). Never let a window cross from one episode
    into another.

    TODO: implement. obs_horizon = act_horizon = 1 is plain behaviour cloning; larger horizons are optional (action
    chunking)."""
    raise NotImplementedError('TODO: il/dataset.py make_pairs')


class Normalizer:
    """Per-dimension standardisation, fitted on the training set only.

    TODO: fit(x) stores the mean and std of x (N, D) (guard std against 0); __call__(x) = (x - mean) / std;
    inverse(y) undoes it; state_dict() / from_state(d) so the checkpoint can carry it (lists or arrays of floats)."""
    def fit(self, x): raise NotImplementedError('TODO: Normalizer.fit')
    def __call__(self, x): raise NotImplementedError('TODO: Normalizer.__call__')
    def inverse(self, y): raise NotImplementedError('TODO: Normalizer.inverse')
    def state_dict(self): raise NotImplementedError('TODO: Normalizer.state_dict')

    @classmethod
    def from_state(cls, d): raise NotImplementedError('TODO: Normalizer.from_state')


class BCDataset:
    """A torch Dataset of normalised (x, y) pairs.

    TODO: subclass torch.utils.data.Dataset (import torch inside, so the sim does not need it); __len__ and
    __getitem__(i) -> (float32 tensor x_i, float32 tensor y_i)."""
    def __init__(self, X, Y, x_norm, y_norm):
        raise NotImplementedError('TODO: BCDataset')
