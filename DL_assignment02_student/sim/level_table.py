#!/usr/bin/env python3
"""The part of Robot_Recording/level_table.py the simulator uses: the tabletop plane the real robot measured by touch
(closed fingertips at 5 points, data/table_probe.json), which robot_env.table_frame turns into the table frame."""
import json
from pathlib import Path

import numpy as np

PROBE = Path(__file__).resolve().parents[1] / 'data' / 'table_probe.json'


def probe_plane(path):
    """(n, d) with n.p + d = height above the touched tabletop, n up."""
    pr = json.loads(Path(path).read_text())
    X = np.array([[p['xy'][0], p['xy'][1], 1.] for p in pr['probes']]); z = np.array([p['table_z'] for p in pr['probes']])
    a, b, c = np.linalg.lstsq(X, z, rcond=None)[0]               # z = a x + b y + c
    n = np.array([-a, -b, 1.]); s = np.linalg.norm(n); return n / s, -c / s
