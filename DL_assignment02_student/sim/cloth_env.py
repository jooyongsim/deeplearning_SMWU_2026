"""Cloth folding with the robot in the loop: UR5e + Robotiq 2F-85 (robosuite's models, as in Assignment 01) above the
same table, and the cloth mesh made from a photo (mesh/photo2mesh.py) as a MuJoCo flex.

The scene (one XML, compiled once per mesh; a new seed only moves the cloth):
  table     the touched tabletop is z = 0 of the table frame (data/table_probe.json, as Assignment 01); a plane geom
            collides with the cloth (MuJoCo 3.3.1: one contact per cloth vertex), a box under it is drawn only
  cloth     flexcomp "direct": the mesh's vertices and triangles, its photo as the texture. Every vertex is a small
            body with 3 slide joints; edge equality constraints keep the triangle edges at their rest length
            (inextensible woven cotton); no bending stiffness, no self-collision (layers may overlap, see README)
  robot     the arm is driven kinematically, as Assignment 01: the action is the gripper's (TCP) velocity, turned into
            joint velocities by damped least squares with the gripper held pointing down, written every physics step
  grasp     closing the gripper attaches the cloth vertices within GRASP_R of the fingertips (at most GRASP_K) to a
            mocap body at the TCP with "connect" equality constraints; opening releases them. (The fingers do not
            collide with anything: a pinch grasp of a thin cloth is not something MuJoCo's contacts do reliably.)

    obs    (80,) cloth keypoints xyz (5 x 5 grid on the cloth, 75) relative to (CENTRE, 0), TCP xyz (3), gripper
           closed (1), number of vertices held / GRASP_K (1)          -> see scene.OBS_LAYOUT
    action [vx, vy, vz, grip] in [-1, 1]: TCP velocity in the TABLE frame x VMAX, grip > 0 closes, <= 0 opens; 10 Hz
    info   fold_metrics.measure (gap_a, gap_b, cover, folds, success), held, blocked, bad
"""
import json, sys, xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fold_metrics as FM
from level_table import PROBE, probe_plane

ROOT = Path(__file__).resolve().parents[1]
PHYS = json.loads((ROOT / 'data' / 'cloth_physics.json').read_text())
START = json.loads((ROOT / 'data' / 'robot_start.json').read_text())

CTRL_DT, DT = 0.1, PHYS['timestep']             # 10 Hz control, physics step
VMAX = 0.25                                     # m/s per axis at |action| = 1
CENTRE = np.array([-0.10, -0.42])               # cloth centre in the table frame (in front of the robot, in reach)
WORK = np.array([[CENTRE[0] - .32, CENTRE[1] - .32, .003], [CENTRE[0] + .32, CENTRE[1] + .32, .30]])   # TCP box
HOME = np.r_[CENTRE, .15]                       # TCP at the start of an episode
GRASP_R, GRASP_K = 0.03, 6                      # grasp radius (m) around the TCP, max vertices held
FINGER = START['finger_extent_m']               # flange -> closed fingertips (m)
GRIP_OPEN, GRIP_CLOSED = .25, .78               # Robotiq finger joint (0 open .. 0.8 closed), drawn only
TABLE, TABLE_H = (-0.46, -0.92, 0.96, 0.82), 0.75
ARM_J = [f'robot0_{j}_joint' for j in ('shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3')]
FINGER_J = [('gripper0_finger_joint', 1), ('gripper0_left_inner_knuckle_joint', 1), ('gripper0_left_inner_finger_joint', -1),
            ('gripper0_right_outer_knuckle_joint', 1), ('gripper0_right_inner_knuckle_joint', 1), ('gripper0_right_inner_finger_joint', -1)]


def table_frame(probe=PROBE):
    n, d = probe_plane(probe)
    x = np.array([1., 0, 0]) - n[0] * n; x /= np.linalg.norm(x); y = np.cross(n, x)
    T = np.eye(4); T[:3, :3] = np.stack([x, y, n]); T[:3, 3] = [0, 0, d]; return T


def quat(R):
    q = np.empty(4); mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).ravel()); return q


def load_mesh(mesh_dir):
    """The cloth mesh (mesh/photo2mesh.py's cloth_mesh.npz + texture.png)."""
    z = np.load(Path(mesh_dir) / 'cloth_mesh.npz')
    return dict(V=z['vertices'].astype(float), F=z['faces'].astype(np.int32), uv=z['uv'].astype(float), corners=z['corners'].astype(int),
                width=float(z['width_m']), height=float(z['height_m']), texture=Path(mesh_dir).resolve() / 'texture.png')


def keypoint_ids(uv, n=5):
    """Vertices nearest to an n x n grid in texture coordinates, row by row from corner c0 (u = v = 0)."""
    g = np.linspace(0, 1, n); return np.array([np.argmin(np.linalg.norm(uv - [u, v], axis=1)) for v in g for u in g])


class ClothFoldEnv:
    def __init__(self, mesh_dir, horizon=1200):
        self.mesh = load_mesh(mesh_dir); self.horizon = horizon
        M = self.mesh; self.N = len(M['V']); self.kp_ids = keypoint_ids(M['uv'])
        a, b = M['V'][M['F'][:, 1]] - M['V'][M['F'][:, 0]], M['V'][M['F'][:, 2]] - M['V'][M['F'][:, 0]]
        self.flat_area = float(.5 * np.abs(np.cross(a[:, :2], b[:, :2])).sum())
        self.T_tb = table_frame(); self.R_tb = self.T_tb[:3, :3]
        self.m = mujoco.MjModel.from_xml_string(self.xml(), {'cloth_texture.png': self.mesh['texture'].read_bytes()});   # in memory: MuJoCo cannot open non-ASCII paths
        self.d = mujoco.MjData(self.m); self.kd = mujoco.MjData(self.m)
        m = self.m
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_J]
        self.qadr, self.vadr, self.lim = m.jnt_qposadr[jid], m.jnt_dofadr[jid], m.jnt_range[jid]
        self.fadr = [(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)], m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)], c) for j, c in FINGER_J]
        self.hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'robot0_right_hand')
        self.eq = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f'grasp{k}') for k in range(self.N)])
        vb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'cloth_{k}') for k in range(self.N)]
        self.vq = np.array([m.jnt_qposadr[m.body_jntadr[b]] for b in vb])           # each vertex: 3 slide joints x, y, z
        self.vd = np.array([m.jnt_dofadr[m.body_jntadr[b]] for b in vb])
        self.vpos0 = np.array([m.body_pos[b] for b in vb])                          # the vertex bodies' rest positions
        self.cloth_dof = np.concatenate([np.arange(a, a + 3) for a in self.vd])
        self.R_down = self.R_tb @ np.array(START['R_flange_base'])                  # gripper orientation held all episode (table frame)
        self.rng = np.random.default_rng(0)
        self.rot_max, self.off_max = np.radians(30), .03

    # ---- model ----
    def xml(self):
        from robosuite.models import MujocoWorldBase
        from robosuite.models.grippers import gripper_factory
        from robosuite.models.robots import UR5e
        world = MujocoWorldBase(); robot = UR5e(); robot.add_gripper(gripper_factory('Robotiq85Gripper')); robot.set_base_xpos([0, 0, 0])
        world.merge(robot); root = ET.fromstring(world.get_xml())
        base = next(b for b in root.iter('body') if b.get('name') == 'robot0_base')
        base.set('quat', ' '.join(f'{v:.6f}' for v in quat(self.R_tb @ np.diag([-1., -1., 1.]))))     # UR base: half turn about z
        base.set('pos', ' '.join(f'{v:.6f}' for v in self.T_tb[:3, 3]))
        for g in root.iter('geom'):
            g.attrib.update(contype='0', conaffinity='0')
            if g.get('group', '0') == '0': g.set('rgba', '0 0 0 0')         # collision meshes: not drawn
        for s in root.iter('site'): s.set('rgba', '0 0 0 0')
        for t in root.findall('tendon'): root.remove(t)
        for o in root.findall('option'): root.remove(o)
        for a in root.findall('actuator'): root.remove(a)                   # arm and fingers are written kinematically
        P, M = PHYS, self.mesh; wb = root.find('worldbody'); b = self.T_tb[:3, 3]
        ET.SubElement(wb, 'geom', type='cylinder', fromto=f'{b[0]:.4f} {b[1]:.4f} {-TABLE_H:.4f} {b[0]:.4f} {b[1]:.4f} {b[2] - .001:.4f}',
                      size='0.09', rgba='.35 .35 .37 1', contype='0', conaffinity='0', group='1')
        x0, y0, w, h = TABLE
        ET.SubElement(wb, 'geom', name='table_body', type='box', size=f'{w / 2:.4f} {h / 2:.4f} {TABLE_H / 2 - .001:.4f}',
                      pos=f'{x0 + w / 2:.4f} {y0 + h / 2:.4f} {-TABLE_H / 2 - .001:.4f}', material='wood', contype='0', conaffinity='0', group='1')
        ET.SubElement(wb, 'geom', name='table_top', type='plane', size=f'{w / 2:.4f} {h / 2:.4f} .01', pos=f'{x0 + w / 2:.4f} {y0 + h / 2:.4f} 0',
                      material='wood', friction=f'{P["mu_table"]} .005 .0001', contype='1', conaffinity='1')
        ET.SubElement(wb, 'geom', name='floor', type='plane', size='3 3 .1', pos=f'0 0 {-TABLE_H}', rgba='.55 .55 .57 1', contype='0', conaffinity='0')
        ET.SubElement(wb, 'light', pos=f'{CENTRE[0]} {CENTRE[1]} 1.6', dir='0 0 -1', diffuse='.75 .75 .75')
        ET.SubElement(wb, 'light', pos=f'{CENTRE[0] + .8} {CENTRE[1] - .8} 1.2', dir='-.5 .5 -1', diffuse='.3 .3 .3')
        g = ET.SubElement(wb, 'body', name='grip', mocap='true', pos=' '.join(map(str, HOME)))
        ET.SubElement(g, 'geom', type='sphere', size='.004', rgba='1 .2 .2 0', contype='0', conaffinity='0', group='2')
        r = P['radius']; V = M['V']
        fc = ET.SubElement(wb, 'flexcomp', name='cloth', type='direct', dim='2', radius=str(r), pos='0 0 0', mass=str(P['mass']), material='cloth',
                           point=' '.join(f'{v:.5f}' for v in (V + [CENTRE[0], CENTRE[1], r]).ravel()),
                           element=' '.join(map(str, M['F'].ravel())), texcoord=' '.join(f'{v:.5f}' for v in M['uv'].ravel()))
        ET.SubElement(fc, 'contact', condim='3', solref=f'{2 * DT} 1', selfcollide=P.get('selfcollide', 'none'), internal=P.get('internal', 'false'), friction=str(P['mu_table']), contype='1', conaffinity='1')
        ET.SubElement(fc, 'edge', equality='true', solref=f'{2 * DT} 1', damping=str(P['edge_damping']))
        asset = root.find('asset')
        if asset is None: asset = ET.SubElement(root, 'asset')
        ET.SubElement(asset, 'texture', name='cloth', type='2d', file='cloth_texture.png')
        ET.SubElement(asset, 'material', name='cloth', texture='cloth', rgba='1 1 1 1', specular='0', shininess='0')
        ET.SubElement(asset, 'texture', name='wood', type='2d', builtin='flat', rgb1='.72 .55 .38', rgb2='.6 .45 .3', mark='random', random='.02', width='512', height='512')
        ET.SubElement(asset, 'material', name='wood', texture='wood', texrepeat='4 4', reflectance='0.02')
        eq = root.find('equality')
        if eq is None: eq = ET.SubElement(root, 'equality')
        for k in range(self.N):
            ET.SubElement(eq, 'connect', name=f'grasp{k}', body1=f'cloth_{k}', body2='grip', anchor='0 0 0', active='false', solref=f'{2 * DT} 1')
        ET.SubElement(root, 'option', timestep=str(DT), solver='CG', tolerance='1e-4', iterations='30', integrator='implicitfast')
        ET.SubElement(root, 'size', memory='200M')
        vis = root.find('visual')
        if vis is None: vis = ET.SubElement(root, 'visual')
        ET.SubElement(vis, 'global', offwidth='1280', offheight='960')
        ET.SubElement(vis, 'quality', shadowsize='4096')
        return ET.tostring(root, encoding='unicode')

    # ---- kinematics (as Assignment 01) ----
    def fk(self, q):
        self.kd.qpos[self.qadr] = q; mujoco.mj_kinematics(self.m, self.kd); mujoco.mj_comPos(self.m, self.kd)
        return self.kd.xpos[self.hand].copy(), self.kd.xmat[self.hand].reshape(3, 3).copy()

    def tcp_of(self, q):
        p, R = self.fk(q); return p + R[:, 2] * FINGER

    def dls(self, q, dp, R):
        p, Rc = self.fk(q); e_r = .5 * sum(np.cross(Rc[:, i], R[:, i]) for i in range(3))
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv)); mujoco.mj_jacBody(self.m, self.kd, jp, jr, self.hand)
        J = np.vstack([jp[:, self.vadr], jr[:, self.vadr]])
        return J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.r_[dp, e_r])

    def ik_tcp(self, q, tcp, iters=50):
        """Joint angles putting the TCP at tcp with the gripper pointing down."""
        for _ in range(iters):
            dq = self.dls(q, tcp - self.tcp_of(q), self.R_down); q = np.clip(q + dq, self.lim[:, 0], self.lim[:, 1])
            if np.linalg.norm(dq) < 1e-7: break
        return q

    def set_robot(self, q, qd, g):
        self.d.qpos[self.qadr] = q; self.d.qvel[self.vadr] = qd
        for a, v, c in self.fadr: self.d.qpos[a] = c * g; self.d.qvel[v] = 0

    # ---- cloth state ----
    def verts(self):
        """Cloth vertex positions (N, 3), table frame."""
        return self.vpos0 + self.d.qpos[self.vq[:, None] + np.arange(3)]

    def set_verts(self, X):
        self.d.qpos[self.vq[:, None] + np.arange(3)] = X - self.vpos0; self.d.qvel[self.vd[:, None] + np.arange(3)] = 0

    # ---- episode ----
    def reset(self, seed=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.m, self.d)
        a = self.rng.uniform(-self.rot_max, self.rot_max); r_, t_ = self.off_max * np.sqrt(self.rng.random()), self.rng.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]); self.start_tf = (float(np.degrees(a)), [r_ * np.cos(t_), r_ * np.sin(t_)])
        V = self.mesh['V'][:, :2] @ R.T + CENTRE + self.start_tf[1]
        self.set_verts(np.c_[V, np.full(len(V), PHYS['radius'])])
        self.q = self.ik_tcp(np.array(START['q'], float), HOME); self.g = GRIP_OPEN; self.closed = False; self.held = np.zeros(0, int)
        self.d.eq_active[self.eq] = 0; self.set_robot(self.q, np.zeros(6), self.g); self.d.mocap_pos[0] = self.tcp_of(self.q)
        mujoco.mj_forward(self.m, self.d)
        self.t, self.blocked, self.last_a = 0, 0, np.zeros(4)
        self.info = dict(FM.measure(self.verts()[:, :2], self.mesh['F'], self.mesh['corners'], self.flat_area), held=0, blocked=False, bad=False)
        return self.obs()

    def tcp(self): return self.tcp_of(self.q)

    def keypoints(self): return self.verts()[self.kp_ids]

    def obs(self):
        kp = self.keypoints() - np.r_[CENTRE, 0]
        return np.concatenate([kp.ravel(), self.tcp() - np.r_[CENTRE, 0], [float(self.closed), len(self.held) / GRASP_K]]).astype(np.float32)

    def grasp(self):
        X = self.verts(); p = self.tcp(); dist = np.linalg.norm(X - p, axis=1)
        near = np.argsort(dist)[:GRASP_K]; near = near[dist[near] < GRASP_R]
        for k in near:
            e = self.eq[k]; self.m.eq_data[e, 0:3] = 0; self.m.eq_data[e, 3:6] = X[k] - p; self.d.eq_active[e] = 1
        self.held = near

    def release(self):
        self.d.eq_active[self.eq] = 0; self.held = np.zeros(0, int)

    def step(self, a):
        a = np.clip(np.asarray(a, float), -1, 1); self.last_a = a.copy()
        close = a[3] > 0
        if close and not self.closed: self.closed = True; self.grasp()
        elif not close and self.closed: self.closed = False; self.release()
        p0 = self.tcp(); want = np.clip(p0 + a[:3] * VMAX * CTRL_DT, WORK[0], WORK[1])
        q1 = self.q.copy()
        for _ in range(8): q1 = np.clip(q1 + self.dls(q1, want - self.tcp_of(q1), self.R_down), self.lim[:, 0], self.lim[:, 1])
        p1 = self.tcp_of(q1); n = round(CTRL_DT / DT); qd = (q1 - self.q) / CTRL_DT; q0, g0 = self.q, self.g
        g1 = GRIP_CLOSED if self.closed else GRIP_OPEN
        for s in range(n):
            u = (s + 1) / n; self.set_robot(q0 + (q1 - q0) * u, qd, g0 + (g1 - g0) * min(1, 3 * u)); self.d.mocap_pos[0] = p0 + (p1 - p0) * u
            mujoco.mj_step(self.m, self.d)
        self.q, self.g = q1, g1; self.t += 1
        miss = np.linalg.norm(p1 - want) > .01; self.blocked = self.blocked + 1 if miss else 0
        bad = not np.isfinite(self.d.qpos).all() or np.abs(self.d.qvel[self.cloth_dof]).max() > 100
        X = self.verts()
        self.info = dict(FM.measure(X[:, :2], self.mesh['F'], self.mesh['corners'], self.flat_area), held=len(self.held), blocked=bool(miss), bad=bool(bad))
        done = bad or self.t >= self.horizon
        return self.obs(), 0., bool(done), dict(self.info)
