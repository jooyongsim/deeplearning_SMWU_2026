"""Milestone 3 (b): what the policy is rewarded for.                                              TODO

Each RL step is one fold. After it, the simulator measures (sim/fold_metrics.py):
    gap_a, gap_b   distance between the corners of each diagonal pair (mm): 424 at the start, ~0-70 when folded
    cover          area covered seen from above / flat area: 1 spread out, ~0.5 after one fold, ~0.25 after two
    folds          corner pairs closed (gap <= 70 mm): 0, 1, 2
    success        both pairs closed and cover <= 0.35 (the task)
    bad            the simulation broke (it should not happen; punish it)
"""


def compute_reward(prev, info, k, done, info0):
    """prev / info: the measurement before / after this fold (dicts as above); k: folds done including this one;
    done: True on the episode's last step; info0: the measurement at the start of the episode.

    Requirements (python rl/check_rl.py tests them):
      - a fold that brings a corner pair together earns clearly more than a fold that changes nothing
      - doing nothing earns about 0, making it worse earns less than doing nothing
      - success at the end earns more than ending at the same gaps without success
      - a broken simulation (info['bad']) is punished (< -0.5)
    Think about: a reward only at success is sparse (a random policy almost never succeeds); a reward for each step's
    progress is dense. Potential-based shaping, r = phi(after) - phi(before), keeps the best policy the same.
    """
    # TODO: compute the reward
    raise NotImplementedError('rl/reward.py: compute_reward')
