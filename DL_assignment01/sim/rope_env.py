"""MuJoCo rope-T pushing environment with the fitted material (fit_t.json), for imitation then reinforcement learning.

Same rope model as Hand_Material_Estimation/mujoco_cable/fit_t.py (robot mode): a tree of 16 mm capsule links on ball
joints, free on the table (z = 0, the touched tabletop), grown from the junction along three branches of the real
lengths. The robot is reduced to what touches the rope in the demos: the pusher stick, a vertical capsule whose tip
the policy moves (welded to a mocap target, so friction acts as for a moving body).

    obs    rope keypoints (xy, at fixed arcs along arm, arm, stem), stick tip xyz, goal keypoints - table frame, metres
    action stick tip velocity in [-1, 1]^3, scaled to VMAX (xy) / VZ (z); 10 Hz
    reward progress: 20 x the decrease of the mean keypoint distance to the goal (arms matched either way round) per
           step, +1 at success (< SUCCESS m), -1 if the simulation blows up. (Run 1 rewarded -distance: the rope
           barely moves by chance, so PPO learned to leave it alone.)
"""
import json
from pathlib import Path

import os
import mujoco
import numpy as np

import success as SC

FIT = Path(os.environ.get('ROPE_FIT', Path(__file__).resolve().parents[1] / 'data' / 'fit_t.json'))   # the material fitted on the real demos (ROPE_FIT: another one)
LINK, DT, MASS, RADIUS = 0.016, 0.002, 0.023, 0.01     # rope-T weighed 23 g (2026-10-09)
LENGTHS = np.array([0.085, 0.085, 0.16])
ARCS = [np.linspace(0, LENGTHS[0], 5), np.linspace(0, LENGTHS[1], 5), np.linspace(0, LENGTHS[2], 9)]
CTRL_DT, VMAX, VZ, STICK_R = 0.1, 0.16, 0.08, 0.005      # demos reach 150 mm/s with Shift; stick rod 10 mm (hardware/gripper_probe)
SUCCESS = 0.02      # the old criterion (mean keypoint distance); success is now success.is_success (position, rotation, T shape)
# The table: its top is the touched tabletop (z = 0); extent (x0, y0, width, depth) from the cameras' tabletop points,
# 2026-10-09 (the robot base stands ~10 cm behind the near edge, y = -0.10); its height above the floor is assumed.
TABLE, TABLE_H = (-0.46, -0.92, 0.96, 0.82), 0.75
CENTRE = np.array([-0.15, -0.45])       # middle of the demos' workspace (table frame); observations are relative to it


def rope_mu(p):
    """The rope geoms' own friction. MuJoCo takes the larger of two geoms' frictions, so with separate table and
    stick frictions (mu_table, mu_stick) the rope's is set below both and each contact gets its own value."""
    return .05 if 'mu_table' in p else p['friction']


def frame_of(d):
    x = d / np.linalg.norm(d); z = np.array([0., 0., 1.]) - x[2] * x; z /= np.linalg.norm(z); return np.stack([x, np.cross(z, x), z], 1)


def quat(R):
    q = np.empty(4); mujoco.mju_mat2Quat(q, R.ravel()); return q


def branch_points(junction, branch, length):
    """Model link endpoints along an observed branch polyline: the real length, the observed direction."""
    pl = branch[np.isfinite(branch[:, 0]), :2]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pl, axis=0), axis=1))]; ratio = s[-1] / length
    k = int(round(length / LINK)); arcs = np.linspace(0, length, k + 1)
    xy = np.stack([np.interp(arcs * ratio, s, pl[:, j]) for j in (0, 1)], 1)
    q = [np.r_[junction[:2], RADIUS]]
    for i in range(1, len(arcs)):
        d = np.r_[xy[i] - xy[i - 1], 0]; d /= max(np.linalg.norm(d), 1e-9); q.append(q[-1] + d * length / k)
    return np.array(q)


def keypoints(segs_per_branch):
    """xy at ARCS along each branch from its chain of link segments [(start, end), ...]."""
    out = []
    for b, segs in enumerate(segs_per_branch):
        p = np.vstack([segs[0][0]] + [e for _, e in segs])[:, :2]
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        out.append(np.stack([np.interp(ARCS[b], s, p[:, j]) for j in (0, 1)], 1))
    return np.concatenate(out)


def keypoints_z(segs_per_branch):
    """Height above the table of the rope's axis at the keypoints (19, the same ARCS as keypoints()): 3-D keypoints'
    z. A rope lying flat has its axis at RADIUS; higher where it rides on something or is lifted."""
    out = []
    for b, segs in enumerate(segs_per_branch):
        p = np.vstack([segs[0][0]] + [e for _, e in segs])
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1))]
        out.append(np.interp(ARCS[b], s, p[:, 2]))
    return np.concatenate(out)


def goal_distance(kp, goal):
    """Mean keypoint distance, the two arms matched either way round."""
    n0, n1 = len(ARCS[0]), len(ARCS[1])
    swap = np.r_[np.arange(n0, n0 + n1), np.arange(n0), np.arange(n0 + n1, len(kp))]
    return min(np.linalg.norm(kp - goal, axis=1).mean(), np.linalg.norm(kp[swap] - goal, axis=1).mean())


class RopeTEnv:
    def __init__(self, shapes, goal, fit=FIT, horizon=700, perturb=True, seed=0, tips=None):
        """shapes: [(junction, branches)] initial rope shapes (one is drawn per episode); goal: (junction, branches)."""
        self.p = dict(fit) if isinstance(fit, dict) else json.loads(Path(fit).read_text()); self.shapes, self.horizon, self.perturb = shapes, horizon, perturb
        self.scale = 1.     # size of the start perturbations (a curriculum grows it from 0)
        self.rot_max, self.off_max, self.off_disc = .17, .03, False     # turn (rad) and offset (m) at scale 1
        self.tips = tips                                   # each demo's first stick tip: episodes start the stick there
        self.rng = np.random.default_rng(seed)
        self.goal = keypoints([[(a, b) for a, b in zip(q[:-1], q[1:])] for q in
                               [branch_points(goal[0], goal[1][b], LENGTHS[b]) for b in range(3)]])
        self.obs_dim, self.act_dim = 2 * len(self.goal) * 2 + 3, 3

    # ---- model ----
    def xml(self, pts, tip):
        p = self.p; total = sum(len(q) - 1 for q in pts); self.links = []
        # the junction knot's mass: MASS / 40 by default; jmass scales it, the links share the rest (the total stays MASS)
        if 'jmass' in p: m_root = MASS / 40 * p['jmass']; m_link = (MASS - m_root) / total
        else: m_root, m_link = MASS / 40, MASS / total            # as fitted so far (the knot on top of MASS)
        # roll: rolling friction of the rope on the table (condim 6; the twisted cotton rope rolls as well as slides)
        rf = f' condim="6" friction="{rope_mu(p):.4f} 0.005 {p["roll"]:.4g}"' if 'roll' in p else f' friction="{rope_mu(p):.4f}"'
        # mu_branch (2026-10-30, per-part properties): each branch's own table friction [b0, b1, stem]; its geoms carry it at
        # priority 1 (over the table's), the stick at priority 2 keeps mu_stick on its contacts; the junction knot takes the mean
        mb = p.get('mu_branch')
        def rfb(b): return f' priority="1" friction="{(np.mean(mb) if b is None else mb[b]):.4f}"' if mb is not None else rf
        # rope_r: the rope's contact radius (default RADIUS, the measured 1 cm); the rope is laid with its axis that high
        rr = p.get('rope_r', RADIUS)
        def chain(points, b, prefix):
            out, closes, R_prev = '', '', np.eye(3)
            for i in range(len(points) - 1):
                d = points[i + 1] - points[i]; R = frame_of(d); L = np.linalg.norm(d); name = f'{prefix}{i}'
                self.links.append((name, b))
                pos = np.zeros(3) if i == 0 else np.array([plen, 0, 0])
                out += (f'<body name="{name}" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" quat="{" ".join(f"{v:.6f}" for v in quat(R_prev.T @ R))}">'
                        f'<joint type="ball" stiffness="{p["k"] * (p.get("kstem", 1.) if b == 2 else 1.):.6g}" damping="{p["damping"]:.6g}"/>'
                        f'<geom type="capsule" fromto="0 0 0 {L:.6f} 0 0" size="{rr}" mass="{m_link:.6g}"{rfb(b)} rgba=".93 .9 .82 1"/>')
                closes += '</body>'; R_prev, plen = R, L
            return out + closes
        tree = ''.join(chain(pts[b], b, f'b{b}_') for b in range(3)); j = pts[0][0]
        at = f'pos="{tip[0]:.5f} {tip[1]:.5f} {tip[2]:.5f}"'
        return f'''<mujoco><compiler angle="radian"/>
          <option timestep="{DT}" integrator="implicitfast" solver="Newton" tolerance="1e-8" cone="elliptic" impratio="10"/>
          <visual><global offwidth="960" offheight="720"/></visual>
          <worldbody><light pos="0 -0.4 1.5" dir="0 0 -1"/>
          <geom name="floor" type="plane" size="3 3 .1" pos="0 0 {-TABLE_H}" contype="4" rgba=".55 .55 .57 1"/>
          <geom name="table" type="box" size="{TABLE[2] / 2:.4f} {TABLE[3] / 2:.4f} {TABLE_H / 2}" pos="{(TABLE[0] + TABLE[2] / 2):.4f} {(TABLE[1] + TABLE[3] / 2):.4f} {-TABLE_H / 2}"
                friction="{p.get("mu_table", p["friction"]):.4f}" contype="4" rgba=".75 .45 .25 1"/>
          <body name="root" pos="{j[0]:.5f} {j[1]:.5f} {j[2]:.5f}"><freejoint/>
            <geom type="sphere" size="{rr}" mass="{m_root:.5g}"{rfb(None)} rgba=".93 .9 .82 1"/>{tree}</body>
          <body name="stick_target" mocap="true" {at}/>
          <body name="stick" {at} gravcomp="1"><freejoint/><geom type="capsule" fromto="0 0 {STICK_R} 0 0 0.25" size="{STICK_R}" mass="0.3"
            condim="4" {'priority="2"' if 'mu_branch' in p else 'priority="1"' if 'soft_t' in p else ''} friction="{p.get('mu_stick', p.get('grip', 1.0)):.4g} 0.02 0.0001" solref="{p.get('soft_t', .004):.4g} 1" solimp="0.95 0.99 {p.get('soft_w', .001):.4g}" contype="2" conaffinity="1" rgba=".2 .2 .2 1"/></body>
          </worldbody><equality><weld body1="stick" body2="stick_target" solref="0.004 1"/></equality></mujoco>'''

    def build(self, pts, tip):
        m = mujoco.MjModel.from_xml_string(self.xml(pts, tip))
        ids = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n, _ in self.links}
        # links within three hops overlap permanently: no contacts between them (as fit_t.py)
        par = {b: int(m.body_parentid[b]) for b in range(1, m.nbody)}
        def hops(a, b):
            pa, x, d = {}, a, 0
            while x > 0: pa[x] = d; x = par.get(x, 0); d += 1
            x, d = b, 0
            while x > 0:
                if x in pa: return pa[x] + d
                x = par.get(x, 0); d += 1
            return 99
        bl = list(ids.values())
        ex = ''.join(f'<exclude body1="{mujoco.mj_id2name(m, 1, a)}" body2="{mujoco.mj_id2name(m, 1, b)}"/>'
                     for i, a in enumerate(bl) for b in bl[i + 1:] if hops(a, b) <= 3)
        m = mujoco.MjModel.from_xml_string(self.xml(pts, tip).replace('</worldbody>', f'</worldbody><contact>{ex}</contact>'))
        self.bid = [[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n, b in self.links if b == br] for br in range(3)]
        self.len = [[m.geom_size[m.body_geomadr[i]][1] * 2 for i in ids_] for ids_ in self.bid]
        return m

    # ---- episode ----
    def reset(self, shape=None, tip=None, index=None):
        i = self.rng.integers(len(self.shapes)) if index is None else index; jn, br = self.shapes[i] if shape is None else shape; self.i = i
        pts = [branch_points(jn, br[b], LENGTHS[b]) for b in range(3)]
        if 'rope_r' in self.p: pts = [q + [0, 0, self.p['rope_r'] - RADIUS] for q in pts]
        if self.perturb and shape is None:          # a few cm and degrees around the demo starts
            a = self.rng.uniform(-self.rot_max, self.rot_max) * self.scale; R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            if self.off_disc:       # uniform in the disc of radius off_max (not the square)
                r_, t_ = self.off_max * np.sqrt(self.rng.random()), self.rng.uniform(0, 2 * np.pi); off = np.r_[r_ * np.cos(t_), r_ * np.sin(t_), 0] * self.scale
            else: off = np.r_[self.rng.uniform(-self.off_max, self.off_max, 2) * self.scale, 0]
            c = pts[0][0]
            pts = [(q - c) @ R.T + c + off for q in pts]; self.start_tf = (R[:2, :2], c[:2].copy(), off[:2])
        else: off = np.zeros(3); self.start_tf = None
        if tip is None and self.tips is not None and shape is None:
            tip = np.r_[self.tips[i][:2] + off[:2] + self.rng.uniform(-.02, .02, 2), max(self.tips[i][2], .003)] if self.perturb else self.tips[i]
        if tip is None: tip = np.r_[pts[0][0][:2] + self.rng.uniform(-.08, .08, 2), .03]
        self.m = self.build(pts, tip); self.d = mujoco.MjData(self.m); mujoco.mj_forward(self.m, self.d)
        self.target = np.array(tip, float); self.t = 0
        o = self.obs(); self.prev = goal_distance(self.kp, self.goal); return o

    def rope_kp(self):
        segs = []
        for ids, ls in zip(self.bid, self.len):
            x = self.d.xpos[ids]; R = self.d.xmat[ids].reshape(-1, 3, 3)
            segs.append([(x[i], x[i] + R[i][:, 0] * ls[i]) for i in range(len(ids))])
        self.kp_z = keypoints_z(segs)
        return keypoints(segs)

    def obs(self):
        kp = self.rope_kp(); self.kp = kp
        return np.concatenate([(kp - CENTRE).ravel(), (self.goal - CENTRE).ravel(), self.target - np.r_[CENTRE, 0]]).astype(np.float32)

    def step(self, a):
        a = np.clip(np.asarray(a, float), -1, 1); v = a * np.array([VMAX, VMAX, VZ])
        start = self.target.copy(); self.target = self.target + v * CTRL_DT
        self.target[2] = np.clip(self.target[2], .003, .08); self.target[:2] = np.clip(self.target[:2], CENTRE - .3, CENTRE + .3)
        n = round(CTRL_DT / DT)
        for s in range(n):
            self.d.mocap_pos[0] = start + (self.target - start) * (s + 1) / n
            mujoco.mj_step(self.m, self.d)
        self.t += 1
        o = self.obs()
        bad = not np.isfinite(self.d.qpos).all() or np.abs(self.d.qvel).max() > 50
        dist = goal_distance(self.kp, self.goal); err = SC.errors(self.kp, self.goal); success = SC.is_success(self.kp, self.goal)
        r = 20 * (self.prev - dist) + (1. if success else 0.) - (1. if bad else 0.); self.prev = dist
        done = bad or self.t >= self.horizon
        return o, float(r), bool(done), dict(dist=dist, err=err, success=success, bad=bad)
