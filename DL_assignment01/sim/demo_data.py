"""The real demonstrations the scene is built from (no torch): their rope shapes, the goal T, and the robot's start pose
and stick length for each. load and robot_setup are ropeT_sim_bc/sim/train.py's, without the training parts.
"""
from pathlib import Path

import cv2
import numpy as np

DATA = Path(__file__).resolve().parents[1] / 'data'
DEMOS = DATA / 'demos.npz'
KEYBOARD = DATA / 'keyboard'


def load(path=DEMOS):
    """(D, names, shapes, goal): shapes = [(junction, branches)] per demo, goal = (junction, branches)."""
    D = np.load(path); names = list(D['names'])
    shapes = [(D[f'{n}/junction'], D[f'{n}/branches']) for n in names]
    return D, names, shapes, (D['goal/junction'], D['goal/branches'])


def robot_setup(D, names):
    """Each demo's first joint angles and flange rotation (robot start pose), and its stick length."""
    recs = [np.load(KEYBOARD / n / 'real_demo.npz') for n in names]
    return dict(starts=[(np.unwrap(r['q'], axis=0)[0], cv2.Rodrigues(r['tcp'][0, 3:])[0]) for r in recs],
                sticks=[float(D[f'{n}/stick']) for n in names])
