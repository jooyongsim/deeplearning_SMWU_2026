"""Milestone 3 (part 2): the policy network and the agent that runs it in the simulator.

Given (do not change):  save_checkpoint, load_agent, the Agent interface (reset, act)
TODO (you):             MLPPolicy, Agent.act

Checkpoint (torch.save dict): model (state_dict), config (the MLPPolicy kwargs + obs_horizon, act_horizon, ...),
x_norm / y_norm (Normalizer.state_dict()). eval_policy.py only calls load_agent(path) -> Agent.
"""
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from dataset import make_features, Normalizer


class MLPPolicy(nn.Module):
    """features window (B, in_dim) -> actions (B, out_dim), normalised.

    TODO: an MLP (e.g. 2-3 hidden layers of 256, ReLU or GELU). Optional: dropout, LayerNorm, residual blocks. The
    config dict saved with the checkpoint must be enough to rebuild it: MLPPolicy(**config['model'])."""
    def __init__(self, in_dim, out_dim, hidden=256, layers=3):
        super().__init__()
        raise NotImplementedError('TODO: il/policy.py MLPPolicy.__init__')

    def forward(self, x):
        raise NotImplementedError('TODO: il/policy.py MLPPolicy.forward')


class Agent:
    """What eval_policy.py runs: reset() at an episode's start, act(obs (85,)) -> action (3,) in [-1, 1] each step."""
    def __init__(self, model, config, x_norm, y_norm):
        self.model, self.cfg, self.x_norm, self.y_norm = model.eval(), config, x_norm, y_norm
        self.To, self.Ta = int(config.get('obs_horizon', 1)), int(config.get('act_horizon', 1))
        self.hist, self.plan = deque(maxlen=self.To), []

    def reset(self):
        self.hist.clear(); self.plan = []

    @torch.no_grad()
    def act(self, obs):
        """TODO: features of obs (make_features on obs[None]), push into self.hist (fill it with copies at the first step),
        normalise the window as in training, run the model, un-normalise, and return the next action (3,) clipped to
        [-1, 1]. With act_horizon > 1 you may keep the predicted chunk in self.plan and replan when it is used up."""
        raise NotImplementedError('TODO: il/policy.py Agent.act')


# ---------------------------------------------------------------- given
def save_checkpoint(path, model, config, x_norm, y_norm, **extra):
    torch.save(dict(model=model.state_dict(), config=config, x_norm=x_norm.state_dict(), y_norm=y_norm.state_dict(), **extra), path)


def load_agent(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = MLPPolicy(**ck['config']['model']); model.load_state_dict(ck['model'])
    return Agent(model, ck['config'], Normalizer.from_state(ck['x_norm']), Normalizer.from_state(ck['y_norm']))
