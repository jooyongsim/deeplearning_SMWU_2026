"""rope-T pushing with the robot in the loop: UR5e + Robotiq 2F-85 (robosuite's models) holding the pusher stick.

The rope, the table and the reward are RopeTEnv's (fitted material, touched tabletop as z = 0, progress reward). The
robot base sits at its real pose in that table frame (touch probe). The stick that touches the rope is RopeTEnv's (a body
welded to a target, the contact model the material fit and the stick-only runs used); its target is the tip of the
stick the arm holds (forward kinematics, the demo's stick length), drawn in the hand. The arm is driven kinematically, as the real UR
follows speedL: the policy's action is the TCP linear velocity in the robot base frame (the command the real robot
takes), turned into joint velocities by damped least squares on the flange (orientation held at the episode's start)
and integrated over the control step; joint positions and velocities are written every physics step, so the stick
pushes with its real velocity.

    obs    rope keypoints, goal keypoints (table frame, as RopeTEnv), stick tip xyz (table frame), joint angles / pi
    action TCP velocity in the base frame, [-1, 1]^3 -> VMAX (xy) / VZ (z)
    reward RopeTEnv's progress reward, + 5 x the decrease of the stick tip's distance to the rope, -0.05 per step the joints or the table stop the commanded motion
           (IK residual > 1 cm, the stick tip pushed below the table), -1 and the episode ends if that persists 1 s
"""
import xml.etree.ElementTree as ET
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
from robosuite.models import MujocoWorldBase
from robosuite.models.grippers import gripper_factory
from robosuite.models.robots import UR5e

from rope_env import RopeTEnv, CTRL_DT, DT, VMAX, VZ, goal_distance, SUCCESS, CENTRE, TABLE_H
import success as SC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from level_table import PROBE, probe_plane

FOLLOWERS = [('gripper0_left_inner_knuckle_joint', 1), ('gripper0_left_inner_finger_joint', -1), ('gripper0_right_outer_knuckle_joint', 1),
             ('gripper0_right_inner_knuckle_joint', 1), ('gripper0_right_inner_finger_joint', -1)]
ARM_J = [f'robot0_{j}_joint' for j in ('shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3')]
GRIP = .63    # Robotiq finger joint (0 open .. 0.8 closed) holding the stick: 21 % open, as the real gripper (opening 0.212)
STICK_R = 0.005                       # 10 mm rod (hardware/gripper_probe/probe.py)
WORK = np.array([[-.385, -.686, .002], [.168, -.271, .045]])    # stick-tip box (table frame): the demos' tip extent + 8 cm, z up to 45 mm
                                      # (the demos start at 31-43 mm and push at the table); a cloned policy drifted to 29 cm


def table_frame(probe=PROBE):
    n, d = probe_plane(probe)
    x = np.array([1., 0, 0]) - n[0] * n; x /= np.linalg.norm(x); y = np.cross(n, x)
    T = np.eye(4); T[:3, :3] = np.stack([x, y, n]); T[:3, 3] = [0, 0, d]; return T


def quat(R):
    q = np.empty(4); mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).ravel()); return q


def warp_path(path, tf):
    """A demo's stick path for a rope start moved by tf = (2x2 rotation, centre, offset) (RopeTEnv.start_tf): the
    start moved with the rope, blending back to the demo's own path by its end, since the goal stays where it is."""
    if tf is None: return path
    R, c, off = tf; w = np.linspace(1., 0., len(path))[:, None]
    moved = (path[:, :2] - c) @ R.T + c + off
    return np.c_[w * moved + (1 - w) * path[:, :2], path[:, 2]]


class RobotRopeEnv(RopeTEnv):
    def __init__(self, shapes, goal, starts, stick=0.33, sticks=None, mid=None, p_mid=0., paths=None, res_scale=.5,
                 end_on_success=False, kp3d=False, flat_z=None, **kw):
        """starts: per shape, (joint angles, flange rotation in the base frame) the demo began with; sticks: per shape,
        the stick length of that demo (else `stick`)."""
        super().__init__(shapes, goal, **kw); self.starts, self.stick, self.sticks = starts, stick, sticks
        # mid: per demo, a list of (rope shape, joints, flange rotation) at moments through it (track_shapes.py, once a
        # second); with probability p_mid an episode starts at one of them instead of the demo's start (reference-state
        # initialisation: the long task is practised from everywhere along it, near the goal too)
        self.mid, self.p_mid = mid, p_mid
        # cur (None = off): curriculum progress in [0, 1]. Starts are drawn from the last `cur` part of a demo's
        # moments (its start counted as the first), so training begins near the goal and moves back to the demo
        # starts; the start perturbations grow with it. 20 % of the episodes start at a demo start all along (30 % at cur = 1).
        self.cur = None
        self.stick_jitter = True  # False: the robot always starts exactly at the demo's start pose
        self.perturb_max = 1.     # start perturbation at cur = 1, in units of RopeTEnv's (3 cm, 10 degrees)
        # approach shaping: + w_approach x the decrease of the stick tip's distance to the rope (down to contact). The
        # rope term alone pays nothing until the stick touches the rope, so from the demo starts the learner had no
        # signal (run8/run9 learnt pushing near the goal and never reached the rope from the start).
        self.w_approach = 5.
        self.T_tb = table_frame(); self.R_tb = self.T_tb[:3, :3]
        # paths (residual mode): per demo, its stick-tip path at 10 Hz. Each episode follows its demo's path from where it
        # starts (demo start or mid-demo moment): the base action is the TCP velocity that keeps the tip on that path
        # (open loop it brought 6 of 10 demos within 24-45 mm), and the policy's action is a correction added to it,
        # res_scale x [-1, 1]. Observations get the base action and the path phase.
        self.paths, self.res_scale, self.ref_d, self.ref_k = paths, res_scale, None, 0
        # end_on_success: the episode ends (terminal) at success with +5 once, instead of +1 every step there (which
        # inflated the values the critic had to fit)
        self.end_on_success = end_on_success
        # kp3d: the rope keypoints are 3-D - their axis heights (19, rope_env.keypoints_z) follow the joint angles in the
        # observation (85 -> 104; the residual's base action and phase stay last). obs_base: the policies' part.
        self.kp3d = kp3d
        # flat_z (m, None = off): the stick tip stays that high above the table and the policies move it in xy only
        # (sim_teleop.py demonstrations without lifting; the action's z is ignored, last_a holds what ran)
        self.flat_z = flat_z
        self.obs_base = self.obs_dim + 6 + (19 if kp3d else 0)
        self.obs_dim = self.obs_base + (4 if paths is not None else 0)

    # ---- model: robosuite robot + RopeTEnv's table and rope, the stick in the hand ----
    def xml(self, pts, tip):
        world = MujocoWorldBase(); robot = UR5e(); robot.add_gripper(gripper_factory('Robotiq85Gripper')); robot.set_base_xpos([0, 0, 0])
        world.merge(robot); root = ET.fromstring(world.get_xml())
        base = next(b for b in root.iter('body') if b.get('name') == 'robot0_base')
        base.set('quat', ' '.join(f'{v:.6f}' for v in quat(self.R_tb @ np.diag([-1., -1., 1.]))))     # half turn about z: UR base
        base.set('pos', ' '.join(f'{v:.6f}' for v in self.T_tb[:3, 3]))
        for g in root.iter('geom'):
            g.attrib.update(contype='0', conaffinity='0')
            if g.get('group', '0') == '0': g.set('rgba', '0 0 0 0')   # robot collision meshes: mass only (inertiagrouprange 0), not drawn
        for site in root.iter('site'): site.set('rgba', '0 0 0 0')       # robosuite's grip/eef marker sites
        for t in root.findall('tendon'): root.remove(t)
        for o in root.findall('option'): root.remove(o)
        # the pedestal the robot stands on, from the floor to its base (drawn only)
        b = self.T_tb[:3, 3]; wb0 = root.find('worldbody')
        ET.SubElement(wb0, 'geom', type='cylinder', fromto=f'{b[0]:.4f} {b[1]:.4f} {-TABLE_H:.4f} {b[0]:.4f} {b[1]:.4f} {b[2] - .001:.4f}',
                      size='0.09', rgba='.35 .35 .37 1', contype='0', conaffinity='0', group='1')
        hand = next(b for b in root.iter('body') if b.get('name') == 'robot0_right_hand')
        ET.SubElement(hand, 'geom', name='stick_drawn', type='capsule', fromto=f'0 0 0.13 0 0 {self.stick - STICK_R:.4f}', size=str(STICK_R),
                      rgba='.15 .15 .15 1', contype='0', conaffinity='0', mass='0.001', group='1')
        env = ET.fromstring(super().xml(pts, tip))
        root.append(env.find('option'))
        wb = root.find('worldbody')
        for el in env.find('worldbody'):
            if el.get('name') == 'stick':                       # RopeTEnv's physical stick, kept (drawn in the hand instead)
                for g in el.iter('geom'): g.set('rgba', '0 0 0 0')
            wb.append(el)
        eq = root.find('equality')
        if eq is None: eq = ET.SubElement(root, 'equality')
        for el in env.find('equality'): eq.append(el)          # the stick's weld to its target
        for j, c in FOLLOWERS:
            ET.SubElement(eq, 'joint', joint1=j, joint2='gripper0_finger_joint', polycoef=f'0 {c} 0 0 0', solref=f'{2 * DT} 1')
        vis = root.find('visual')
        if vis is None: vis = ET.SubElement(root, 'visual')
        ET.SubElement(vis, 'global', offwidth='960', offheight='720')
        return ET.tostring(root, encoding='unicode')

    def build(self, pts, tip):
        m = super().build(pts, tip)
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_J]
        self.qadr, self.vadr = m.jnt_qposadr[jid], m.jnt_dofadr[jid]; self.lim = m.jnt_range[jid]
        self.hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'robot0_right_hand')
        self.fing = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ('gripper0_finger_1', 'gripper0_finger_2')]
        rope = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'root')] + [i for ids in self.bid for i in ids]
        self.rope_dof = np.concatenate([np.arange(m.body_dofadr[b], m.body_dofadr[b] + m.body_dofnum[b]) for b in rope])
        self.kd = mujoco.MjData(m)
        return m

    # ---- kinematics ----
    def fk(self, q):
        self.kd.qpos[self.qadr] = q; mujoco.mj_kinematics(self.m, self.kd); mujoco.mj_comPos(self.m, self.kd)
        return self.kd.xpos[self.hand].copy(), self.kd.xmat[self.hand].reshape(3, 3).copy()

    def tip_of(self, q):
        p, R = self.fk(q); return p + R[:, 2] * self.stick

    def dls(self, q, dp, R):
        """Joint step moving the flange by dp (table frame) and toward orientation R: damped least squares."""
        p, Rc = self.fk(q); e_r = .5 * sum(np.cross(Rc[:, i], R[:, i]) for i in range(3))
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv)); mujoco.mj_jacBody(self.m, self.kd, jp, jr, self.hand)
        J = np.vstack([jp[:, self.vadr], jr[:, self.vadr]])
        return J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.r_[dp, e_r])

    def ik(self, q, p, R, iters=50):
        for _ in range(iters):
            fp, _ = self.fk(q); dq = self.dls(q, p - fp, R); q = np.clip(q + dq, self.lim[:, 0], self.lim[:, 1])
            if np.linalg.norm(dq) < 1e-6: break
        return q

    def place_stick(self, tip):
        """Physical stick (RopeTEnv's welded body) and its target at the tip, then forward."""
        sb = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, 'stick'); adr = self.m.jnt_qposadr[self.m.body_jntadr[sb]]
        self.d.mocap_pos[0] = tip; self.d.qpos[adr:adr + 3] = tip; self.d.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.m, self.d)

    def close_fingers(self, g=GRIP):
        """Fingers already at the holding opening (the actuator target) at the start, instead of closing from open."""
        for name, c in [('gripper0_finger_joint', 1)] + FOLLOWERS:
            j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, name); self.d.qpos[self.m.jnt_qposadr[j]] = c * g

    def set_arm(self, q, qd):
        self.d.qpos[self.qadr] = q; self.d.qvel[self.vadr] = qd; self.d.ctrl[self.fing] = GRIP

    # ---- episode ----
    def reset(self, shape=None, tip=None, start=None, demo=None, k0=0, index=None):
        self.from_start = False; self.ref_d, self.ref_k = demo, k0
        if shape is None and self.cur is not None and self.mid:
            d = int(self.rng.integers(len(self.mid))); m = self.mid[d]; n = len(m) + 1
            k = 0 if self.rng.random() < (.3 if self.cur >= 1 else .2) else int(self.rng.integers(min(int((1 - self.cur) * n), n - 1), n))
            if self.sticks is not None: self.stick = self.sticks[d]
            if k > 0: return self.reset(shape=m[k - 1][0], start=(m[k - 1][1], m[k - 1][2]), demo=d, k0=int(round(m[k - 1][3] * 10)))
            self.scale = self.cur * self.perturb_max; self.ref_d, self.ref_k = d, 0; o = self.reset_demo_start(d); self.from_start = True; return o
        if shape is None and self.mid and self.rng.random() < self.p_mid:
            d = self.rng.integers(len(self.mid)); moments = self.mid[d]
            if moments:
                shape, q, R, tk = moments[self.rng.integers(len(moments))]
                if self.sticks is not None: self.stick = self.sticks[d]
                return self.reset(shape=shape, start=(q, R), demo=d, k0=int(round(tk * 10)))
        i = (self.rng.integers(len(self.shapes)) if index is None else index) if shape is None else 0
        if self.sticks is not None and shape is None: self.stick = self.sticks[i]     # before the model is built
        if shape is None: self.ref_d, self.ref_k = int(i), 0
        return self.reset_demo_start(i, shape, start)

    def reset_demo_start(self, i, shape=None, start=None):
        super().reset(shape=shape, tip=np.zeros(3), index=i)        # rope (perturbed when drawn); self.i = i
        q0, R0 = self.starts[self.i] if start is None else start
        self.R = self.R_tb @ R0                                    # flange orientation held all episode (table frame)
        self.q = np.array(q0, float)
        self.cur_path = None
        if self.paths is not None and self.ref_d is not None:
            self.cur_path = warp_path(self.paths[self.ref_d], self.start_tf) if shape is None else self.paths[self.ref_d]
        if self.perturb and shape is None and self.stick_jitter:    # the stick a little off the (moved) path start
            p, _ = self.fk(self.q); to = self.cur_path[0] - self.tip_of(self.q) if self.cur_path is not None else np.zeros(3)
            self.q = self.ik(self.q, p + to + np.r_[self.rng.uniform(-.02, .02, 2) * min(self.scale, 1.), 0], self.R)
        self.target = self.tip_of(self.q); self.blocked = 0
        self.set_arm(self.q, np.zeros(6)); self.close_fingers(); self.place_stick(self.target)
        o = self.obs(); self.prev = goal_distance(self.kp, self.goal); self.prev_gap = self.gap(); return o

    def gap(self):
        """Stick tip to the nearest rope keypoint (3-D, the rope's axis 10 mm up), no closer than contact. In xy alone
        (run10) the stick earned it hovering at its 35 mm start height above the rope, never touching it."""
        return max(np.linalg.norm(np.c_[self.kp - self.target[:2], np.full(len(self.kp), .01 - self.target[2])], axis=1).min(), .015)

    def obs(self):
        o = super().obs()
        o = np.concatenate([o, np.asarray(getattr(self, 'q', np.zeros(6)), np.float32) / np.pi])
        if self.kp3d: o = np.concatenate([o, self.kp_z])
        if self.paths is not None:
            n = len(self.paths[self.ref_d]) if self.ref_d is not None else 1
            o = np.concatenate([o, self.base_action(), [min(self.ref_k / n, 1.)]])
        return o.astype(np.float32)

    def base_action(self):
        """Residual mode: the TCP velocity command (in [-1, 1]) that brings the tip to the path's next point; 0 past its end."""
        if self.ref_d is None or not hasattr(self, 'R_tb') or not hasattr(self, 'q'): return np.zeros(3)
        p = self.cur_path if getattr(self, 'cur_path', None) is not None else self.paths[self.ref_d]
        if self.ref_k + 1 >= len(p): return np.zeros(3)
        return np.clip(self.R_tb.T @ ((p[self.ref_k + 1] - self.tip_of(self.q)) / CTRL_DT) / np.array([VMAX, VMAX, VZ]), -1, 1)

    def step(self, a):
        a = np.clip(np.asarray(a, float), -1, 1)
        if self.paths is not None: a = np.clip(self.base_action() + self.res_scale * a, -1, 1); self.ref_k += 1
        v_tab = self.R_tb @ (a * np.array([VMAX, VMAX, VZ]))       # base-frame TCP velocity -> table frame
        if self.flat_z is not None:     # z-free mode: the tip kept at flat_z above the table, the action's z ignored
            v_tab[2] = np.clip((self.flat_z - self.tip_of(self.q)[2]) / CTRL_DT, -VZ, VZ)
        self.last_a = np.clip(self.R_tb.T @ v_tab / np.array([VMAX, VMAX, VZ]), -1, 1)     # the action executed
        want = self.tip_of(self.q) + v_tab * CTRL_DT
        want = np.clip(want, WORK[0], WORK[1])     # workspace box: on or above the table, below 45 mm, around the demos
        p, _ = self.fk(self.q); q1 = self.ik(self.q, p + (want - self.tip_of(self.q)), self.R, iters=8)
        qd = (q1 - self.q) / CTRL_DT; n = round(CTRL_DT / DT); q0 = self.q
        t0, t1 = self.tip_of(q0), self.tip_of(q1)
        for s in range(n):
            self.set_arm(q0 + qd * (s + 1) * DT, qd); self.d.mocap_pos[0] = t0 + (t1 - t0) * (s + 1) / n
            mujoco.mj_step(self.m, self.d)
        self.q = q1; self.target = self.tip_of(q1); self.t += 1
        o = self.obs()
        bad = not np.isfinite(self.d.qpos).all() or np.abs(self.d.qvel[self.rope_dof]).max() > 50
        miss = np.linalg.norm(self.target - want) > .01 or self.target[2] < -.003
        self.blocked = self.blocked + 1 if miss else 0
        dist = goal_distance(self.kp, self.goal); err = SC.errors(self.kp, self.goal); success = SC.is_success(self.kp, self.goal)
        stuck = self.blocked >= 10
        gap = self.gap()
        r = 20 * (self.prev - dist) + self.w_approach * (self.prev_gap - gap) \
            + ((5. if self.end_on_success else 1.) if success else 0.) - (1. if bad or stuck else 0.) - (.05 if miss else 0.)
        self.prev, self.prev_gap = dist, gap
        done = bad or stuck or self.t >= self.horizon or (self.end_on_success and success)
        return o, float(r), bool(done), dict(dist=dist, err=err, success=success, bad=bad, blocked=miss, start=self.from_start)
