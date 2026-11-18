"""Milestone 3 (a): what the policy sees.                                                          TODO

The simulator gives an 80-dim observation (sim/scene.py, OBS_LAYOUT):
    obs[0:75]   25 cloth keypoints x (x, y, z), a 5 x 5 grid on the cloth, row by row from corner c0
                (keypoint 0 = c0, 4 = c1, 24 = c2, 20 = c3, 12 = centre); metres, relative to the workspace centre
    obs[75:78]  gripper TCP xyz          obs[78] gripper closed          obs[79] vertices held / 6
Between folds the gripper is always up and open, so the last 5 numbers carry little information.

The policy (an MLP) learns best from inputs of roughly unit scale. The cloth spans about +-0.25 m in x and y; the
heights are a few millimetres (one layer is 4 mm thick, so a fold shows up in z).
"""
import numpy as np

N_FOLDS = 2


def obs_features(obs, k, info):
    """obs: the 80-dim observation; k: folds done so far (0, 1 or 2); info: the simulator's last measurement
    (sim/fold_metrics.py: gap_a, gap_b, cover, folds, success). Returns a 1-D float32 array, the same length for every k.

    Requirements (python rl/check_rl.py tests them):
      - depends on the cloth keypoints
      - tells the policy how many folds are done (the same cloth after fold 1 needs a different action than before it)
      - every entry roughly in [-10, 10]
    Think about: is z useful? should the policy see info's gaps (would a real robot know them)?
    """
    # TODO: build the feature vector
    raise NotImplementedError('rl/features.py: obs_features')
