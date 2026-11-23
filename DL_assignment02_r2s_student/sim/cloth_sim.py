"""The cloth simulator the observations are fitted to (MuJoCo 4.0.1, given in full).

The cloth is the photo mesh (r2s/photo2mesh.py) as a MuJoCo flex:
  - edge equality constraints: the woven cloth does not stretch
  - elasticity elastic2d="bend": bending stiffness, set by `young` (Pa, with a fixed thickness THICK) -- what you fit
  - damping: how fast motion dies out (`damping`) -- fitted too, or kept at the default
  - mass: the real cloth's (grams, measured)
  - air: drag of the still air, per m^2 of cloth: each vertex gets F = -air * (cloth area / N vertices) * v. A light cloth has a large area for its
    weight, so air resistance is about as strong as gravity on it; flex `damping` only damps the deformation, not the
    motion of the whole cloth, so without `air` the simulated cloth keeps swinging and fluttering (measured on T01:
    4x the high-frequency motion of the video; air = 0.85 brought it to the video's level and halved the image loss).
    Time constant (areal density / air) ~ 0.15 s; a flat cloth would fall at g x that ~ 1.5 m/s. 0 = no air (as before)
  - air_model: 'lin' (above: the same drag in every direction, proportional to v; good for slow hand motion) or 'quad':
    the pressure drag of real air on each triangle, only normal to the cloth, F = -air * 1/2 rho Cd A (v.n)|v.n| n
    (rho Cd = RHO_CD), spread over its 3 vertices; there `air` is a multiple of real air (default 1). Use 'quad' for
    fast motion (flinging, throwing): sliding along its own plane the cloth meets almost no air, falling flat it meets a lot.
Two "hands" (mocap bodies) hold the cloth at the grasped vertices with connect constraints and follow the observed
3-D trajectories of the grasped points. Gravity is -z: observations must be expressed in a gravity-aligned frame.

    sim = ClothSim('assets/my_cloth', young=1e4)
    X = sim.rollout(hands, grasp=(c_left, c_right), t=t)     # hands (T, 2, 3) m at times t (s) -> X (T, N, 3) vertex positions
"""
from pathlib import Path

import mujoco
import numpy as np

DT = 0.001              # physics step (s); Euler, as flex elasticity needs
THICK = 0.002           # flex thickness (m) used with `young` (bending stiffness ~ young x THICK^3)
MASS = 0.020            # cloth mass (kg); ClothSim reads it when constructed, so `cloth_sim.MASS = ...` changes every simulation
AIR = 0.85              # air drag per m^2 of cloth (N s / m^3); likewise read at construction. 0 = no air
AIR_MODEL = 'lin'       # 'lin' or 'quad' (see above); read at construction
RHO_CD = 1.2 * 1.2      # air density (kg/m^3) x drag coefficient of a plate moving normal to itself
SETTLE = 1.0            # s: the cloth hangs still from the first hand positions before the motion starts
YOUNG_RANGE = (1e2, 3e5)
BOW = 0.01             # m: the initial drape bows this much out of the plane of the hands (breaks the symmetry)


def load_mesh(mesh_dir):
    z = np.load(Path(mesh_dir) / 'cloth_mesh.npz')
    return dict(V=z['vertices'].astype(float), F=z['faces'].astype(np.int32), uv=z['uv'].astype(float), corners=z['corners'].astype(int),
                width=float(z['width_m']), height=float(z['height_m']))


class ClothSim:
    def __init__(self, mesh_dir, young=1e4, damping=0.002, mass=None, grasp_k=3, air=None, air_model=None):
        self.mesh = load_mesh(mesh_dir); self.young, self.damping, self.grasp_k = float(young), float(damping), grasp_k
        self.air_model = AIR_MODEL if air_model is None else air_model
        assert self.air_model in ('lin', 'quad'), air_model
        self.mass = float(MASS if mass is None else mass)
        self.air = float((AIR if self.air_model == 'lin' else 1.) if air is None else air)
        self.m = None

    def setup_air(self, m):
        """Prepare model m (built from self.xml) for the air drag; then advance it with self.step(m, d)."""
        N = len(self.mesh['V'])
        if self.air_model == 'lin': m.dof_damping[:] = self.air * self.mesh['width'] * self.mesh['height'] / N; return   # every dof is a cloth vertex slider
        self._vb = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'cloth_{k}') for k in range(N)])
        self._vd = np.array([m.jnt_dofadr[m.body_jntadr[b]] for b in self._vb])[:, None] + np.arange(3)

    def step(self, m, d):
        """One physics step, with the air drag."""
        if self.air_model == 'quad' and self.air > 0:
            X, U, F = d.flexvert_xpos, d.qvel[self._vd], self.mesh['F']
            n = np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]); a2 = np.linalg.norm(n, axis=1, keepdims=True); n = n / np.maximum(a2, 1e-12)
            vn = (U[F].mean(1) * n).sum(1, keepdims=True)                     # face velocity normal to the face
            f = -self.air * .5 * RHO_CD * (a2 / 2) * vn * np.abs(vn) * n / 3  # per face, a third to each vertex
            fv = np.zeros_like(X)
            for j in range(3): np.add.at(fv, F[:, j], f)
            d.xfrc_applied[self._vb, :3] = fv
        mujoco.mj_step(m, d)

    def xml(self, V):
        M = self.mesh; N = len(V); r = 0.002
        eq = ''.join(f'<connect name="h{h}_{k}" body1="cloth_{k}" body2="hand{h}" anchor="0 0 0" active="false" solref="{2 * DT} 1"/>' for h in (0, 1) for k in range(N))
        return f'''<mujoco><option timestep="{DT}" integrator="Euler" solver="CG" tolerance="1e-5" iterations="40" gravity="0 0 -9.81"/>
<size memory="100M"/><visual><global offwidth="960" offheight="720"/></visual>
<worldbody><light pos="0 0 3" dir="0 0 -1"/>
<body name="hand0" mocap="true"/><body name="hand1" mocap="true"/>
<flexcomp name="cloth" type="direct" dim="2" radius="{r}" mass="{self.mass}" rgba=".95 .95 .95 1"
 point="{' '.join(f'{v:.6f}' for v in V.ravel())}" element="{' '.join(map(str, M['F'].ravel()))}">
 <contact selfcollide="none" contype="0" conaffinity="0"/>
 <edge equality="true" solref="{2 * DT} 1" damping="{self.damping}"/>
 <elasticity young="{self.young}" poisson="0.3" thickness="{THICK}" damping="{self.damping}" elastic2d="bend"/>
</flexcomp></worldbody><equality>{eq}</equality></mujoco>'''

    def initial_pose(self, hands0, grasp):
        """The flat mesh placed in the vertical plane through the two hands: the grasped corners at the hands, the
        rest hanging below (then SETTLE lets it drape)."""
        V = self.mesh['V'][:, :2]; a, b = V[grasp[0]], V[grasp[1]]
        ex = (b - a) / np.linalg.norm(b - a); ey = np.array([-ex[1], ex[0]])
        if ((V - a) @ ey).mean() > 0: ey = -ey                       # the cloth body on the -ey side (hangs down)
        u, v = (V - a) @ ex, (V - a) @ ey                             # u along the held edge, v away from it (<= 0)
        h0, h1 = hands0; d = h1 - h0; d[2] = 0; L = np.linalg.norm(b - a)
        e = d / max(np.linalg.norm(d), 1e-9)                          # horizontal direction hand 0 -> hand 1
        mid = (h0 + h1) / 2
        X = mid + np.outer(u - L / 2, e)                              # true size (the points are the rest shape), centred
        X[:, 2] += v                                                  # hang the rest straight down
        # a slight bow out of the vertical plane (BOW m at the middle): a real cloth is never perfectly flat, and a
        # perfectly flat start stays in its plane forever (by symmetry) where bending stiffness does nothing
        n = np.cross(np.r_[e[:2], 0], [0, 0, 1.]); n /= max(np.linalg.norm(n), 1e-9)
        X += np.outer(BOW * np.sin(np.pi * np.clip(u / L, 0, 1)) * np.clip(-v / max(-v.min(), 1e-9), 0, 1), n)
        return X

    def rollout(self, hands, grasp, t, return_settled=False):
        """hands (T, 2, 3) m in a gravity-aligned frame (z up), at times t (T,) s; grasp = (vertex id held by hand 0,
        by hand 1). Returns the cloth vertex positions (T, N, 3) at those times."""
        hands = np.asarray(hands, float); t = np.asarray(t, float) - t[0]
        X0 = self.initial_pose(hands[0], grasp)
        m = mujoco.MjModel.from_xml_string(self.xml(X0))
        self.setup_air(m)
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        Xc = d.flexvert_xpos.copy()
        for h in (0, 1):
            near = np.argsort(np.linalg.norm(Xc - Xc[grasp[h]], axis=1))[:self.grasp_k]
            for k in near:
                # the grasped corner goes to the hand; its neighbours keep their offset from it (a pinch, not a point)
                e = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f'h{h}_{k}'); m.eq_data[e, :3] = 0; m.eq_data[e, 3:6] = Xc[k] - Xc[grasp[h]]; d.eq_active[e] = 1
            d.mocap_pos[h] = hands[0, h]
        for _ in range(int(SETTLE / DT)): self.step(m, d)
        out = []; T = len(t); step = 0
        for i in range(T):
            while step * DT < t[i]:
                tau = step * DT; j = np.searchsorted(t, tau, side='right') - 1; j = min(max(j, 0), T - 2)
                w = (tau - t[j]) / max(t[j + 1] - t[j], 1e-9); hp = hands[j] + (hands[j + 1] - hands[j]) * np.clip(w, 0, 1)
                d.mocap_pos[0], d.mocap_pos[1] = hp[0], hp[1]; self.step(m, d); step += 1
            out.append(d.flexvert_xpos.copy())
        X = np.array(out)
        if not np.isfinite(X).all(): raise FloatingPointError('simulation blew up (young / damping out of range?)')
        return X


def at_uv(X, mesh, uv):
    """Positions (T, P, 3) of the cloth points at texture coordinates uv (P, 2), by barycentric interpolation on the
    flat mesh: compare simulated cloth points with tracked points that have a known uv."""
    V2, F = mesh['uv'], mesh['F']; uv = np.asarray(uv, float); W = np.zeros((len(uv), 3)); I = np.zeros((len(uv), 3), int)
    A, B, C = V2[F[:, 0]], V2[F[:, 1]], V2[F[:, 2]]
    for p, q in enumerate(uv):
        v0, v1, v2 = B - A, C - A, q - A
        d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1); d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
        den = d00 * d11 - d01 * d01; b1 = (d11 * d20 - d01 * d21) / den; b2 = (d00 * d21 - d01 * d20) / den; b0 = 1 - b1 - b2
        bad = np.minimum(np.minimum(b0, b1), b2); f = int(np.argmax(bad))
        W[p] = [b0[f], b1[f], b2[f]]; I[p] = F[f]
    return np.einsum('pk,tpkd->tpd', W, X[:, I])
