"""A synthetic take with a KNOWN stiffness, to test the pipeline where the answer is known.

The simulated cloth (the photo mesh, textured) held by two hands, filmed by a fixed camera with the floor in view.
make_synth() returns the RGB video and, so that lift + fit can be checked without the perception models, also what a
perfect tracker and a realistically WRONG metric depth model would give:
  tracks   the uv grid projected into the camera (all visible)
  depth    rendered depth x a per-frame scale error drifting in [1.3, 1.7] x a smooth 3.5 % shape error (both as we
           measured for Depth Anything V2 Metric Base on our real cloth video against a RealSense)
  truth    the young and air used, the cloth vertices, the camera
Run the real perception on the RGB video too (the notebook does) to see how much the models add.
"""
import cv2
import numpy as np
import mujoco

from cloth_sim import ClothSim, DT, SETTLE


def hand_motion(T=10., fps=30):
    """Both hands hold the 32 cm edge at 1.1 m, bring it to ~16 cm and back twice, move it a little towards the camera
    and sideways, and a small swing at the end."""
    t = np.arange(0, T, 1 / fps); sep = .23 + .07 * np.cos(2 * np.pi * t / 4)
    cx = .04 * np.sin(2 * np.pi * t / 6); cy = .06 * np.sin(2 * np.pi * t / 5); cz = 1.1 + .03 * np.sin(2 * np.pi * t / 3)
    sw = np.where(t > 8, .05 * np.sin(2 * np.pi * (t - 8) / 1.2), 0.)
    H = np.zeros((len(t), 2, 3)); H[:, 0, 0] = cx - sep / 2 + sw; H[:, 1, 0] = cx + sep / 2 + sw; H[:, :, 1] = cy[:, None]; H[:, :, 2] = cz[:, None]
    return t, H


def gather_tilt_motion(T=17., fps=30, sep_open=.30, sep_closed=.07, tilt_deg=40.):
    """The motion that carries information about bending (see docs): hold the held edge spread, bring the hands together
    (the cloth goes slack and folds), tilt the gathered hands to one side and the other (the bunched cloth bends over under
    its weight: bending against gravity), back to level, spread again; twice (8.5 s each)."""
    k = [(0, sep_open, 0), (1, sep_open, 0), (2.5, sep_closed, 0), (3.5, sep_closed, 0), (5, sep_closed, tilt_deg),
         (6.5, sep_closed, -tilt_deg), (7.5, sep_closed, 0), (8.5, sep_open, 0)]
    k = k + [(a + 8.5, b, c) for a, b, c in k[1:]]
    t = np.arange(0, T, 1 / fps); kt = np.array([a for a, _, _ in k])
    sep = np.interp(t, kt, [b for _, b, _ in k]); ang = np.radians(np.interp(t, kt, [c for _, _, c in k]))
    u = np.stack([np.cos(ang), np.zeros_like(ang), np.sin(ang)], -1); mid = np.array([0., 0., 1.1])
    return t, np.stack([mid - sep[:, None] / 2 * u, mid + sep[:, None] / 2 * u], 1)


MOTIONS = dict(default=hand_motion, gather_tilt=gather_tilt_motion)


def camera_pose(eye=(.15, -1.0, 1.35), target=(0., 0., .95)):
    eye, target = np.array(eye, float), np.array(target, float); f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.]); r /= np.linalg.norm(r); u = np.cross(r, f)
    return eye, np.stack([r, -u, f])          # rows: camera x (right), y (down), z (forward) in world


def make_synth(mesh_dir, young=2e4, air=None, W=960, H=540, fovy=60., T=10., grid=11, seed=0, rgb=True, depth_err=(1.3, 1.7), noise=.035, motion='default'):
    """young, air: the true bending stiffness and air drag (air None = cloth_sim.AIR).
    motion: 'default' (hand_motion) or 'gather_tilt' (gather_tilt_motion; give T=17 for both repetitions)."""
    t, Hn = MOTIONS[motion](T); sim = ClothSim(mesh_dir, young=young, air=air); mesh = sim.mesh
    grasp = (int(mesh['corners'][0]), int(mesh['corners'][3])); eye, Rwc = camera_pose()
    f = H / 2 / np.tan(np.radians(fovy) / 2); K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.]])
    xml = sim.xml(sim.initial_pose(Hn[0], grasp))
    xml = xml.replace('<size memory="100M"/>', f'<size memory="100M"/><visual><global offwidth="{W}" offheight="{H}"/></visual><asset><texture name="c" type="2d" file="c.png"/><material name="c" texture="c"/>'
                      '<texture name="f" type="2d" builtin="checker" rgb1=".45 .42 .40" rgb2=".40 .37 .35" width="512" height="512"/><material name="f" texture="f" texrepeat="10 10"/></asset>')
    xml = xml.replace('<visual><global offwidth="960" offheight="720"/></visual>', '')
    xml = xml.replace('rgba=".95 .95 .95 1"', 'material="c" texcoord="' + ' '.join(f'{v:.5f}' for v in mesh['uv'].ravel()) + '"')
    xml = xml.replace('<worldbody>', '<worldbody><geom type="plane" size="5 5 .1" material="f"/><light pos="1 -2 3" dir="-.3 .6 -1" diffuse=".55 .55 .55"/>'
                      '<geom type="box" pos="0 2.4 1.2" size="4 .02 1.2" rgba=".72 .74 .78 1"/>'
                      f'<camera name="cam" pos="{eye[0]} {eye[1]} {eye[2]}" xyaxes="{Rwc[0][0]} {Rwc[0][1]} {Rwc[0][2]} {-Rwc[1][0]} {-Rwc[1][1]} {-Rwc[1][2]}" fovy="{fovy}"/>')
    xml = xml.replace('<body name="hand0" mocap="true"/><body name="hand1" mocap="true"/>',
                      '<body name="hand0" mocap="true"><geom type="sphere" size=".025" rgba=".85 .65 .55 1" contype="0" conaffinity="0"/></body>'
                      '<body name="hand1" mocap="true"><geom type="sphere" size=".025" rgba=".85 .65 .55 1" contype="0" conaffinity="0"/></body>')
    from pathlib import Path
    m = mujoco.MjModel.from_xml_string(xml, {'c.png': (Path(mesh_dir) / 'texture.png').read_bytes()})
    sim.setup_air(m)                                                                  # the same air drag as ClothSim.rollout
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    Xc = d.flexvert_xpos.copy()
    for h in (0, 1):
        for k in np.argsort(np.linalg.norm(Xc - Xc[grasp[h]], axis=1))[:sim.grasp_k]:
            e = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f'h{h}_{k}'); m.eq_data[e, :3] = 0; m.eq_data[e, 3:6] = Xc[k] - Xc[grasp[h]]; d.eq_active[e] = 1
        d.mocap_pos[h] = Hn[0, h]
    for _ in range(int(SETTLE / DT)): sim.step(m, d)
    rr = mujoco.Renderer(m, H, W); rd = mujoco.Renderer(m, H, W); rd.enable_depth_rendering()
    from cloth_sim import at_uv
    g = np.linspace(0, 1, grid); uv = np.array([(u, v) for v in g for u in g])
    rng = np.random.default_rng(seed); drift = depth_err[0] + (depth_err[1] - depth_err[0]) * (.5 + .5 * np.sin(2 * np.pi * t / 7 + rng.uniform(0, 6)))
    frames_rgb, depths, Xs, step = [], [], [], 0
    for i, ti in enumerate(t):
        while step * DT < ti:
            j = min(int(step * DT * 30), len(t) - 2); w = min(step * DT * 30 - j, 1.); hp = Hn[j] + (Hn[j + 1] - Hn[j]) * w
            d.mocap_pos[0], d.mocap_pos[1] = hp[0], hp[1]; sim.step(m, d); step += 1
        Xs.append(d.flexvert_xpos.copy())
        if rgb: rr.update_scene(d, camera='cam'); frames_rgb.append(rr.render().copy())
        rd.update_scene(d, camera='cam'); z = rd.render().copy()
        # depth model error as measured on our real cloth video: a per-frame scale (drift) and a smooth 3.5 % shape error
        field = cv2.resize(rng.standard_normal((6, 10)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        z = z * drift[i] * (1 + noise * field); depths.append(z[::2, ::2].astype(np.float16))
    Xs = np.array(Xs); Puv = at_uv(Xs, mesh, uv)
    q = (Puv - eye) @ Rwc.T; xy = np.stack([f * q[..., 0] / q[..., 2] + W / 2, f * q[..., 1] / q[..., 2] + H / 2], -1)
    # a point is visible if nothing is in front of it: its camera z vs the clean rendered depth at its pixel
    vis = np.zeros(xy.shape[:2], bool)
    for i in range(len(t)):
        zc = depths[i].astype(np.float32) / drift[i]; px = np.clip(np.round(xy[i] / 2).astype(int), 0, [W // 2 - 1, H // 2 - 1])
        vis[i] = zc[px[:, 1], px[:, 0]] > q[i, :, 2] - .015
    kreg = int(np.argmax(np.linalg.norm(Hn[:, 0] - Hn[:, 1], axis=1)[:90]))
    qc = (Xs[kreg][mesh['corners']] - eye) @ Rwc.T; C = np.stack([f * qc[:, 0] / qc[:, 2] + W / 2, f * qc[:, 1] / qc[:, 2] + H / 2], -1)
    corner_ids = np.array([int(np.argmin(np.linalg.norm(uv - c, axis=1))) for c in [[0, 0], [1, 0], [1, 1], [0, 1]]])
    frames = np.arange(len(t))
    return dict(video=np.array(frames_rgb) if rgb else None, t=t, K=K, W=W, H=H,
                reg=dict(frame=kreg, corners_px=C.tolist(), held=[0, 3], start=0, end=len(t) - 1, fit_end=int(len(t) * .6)),
                tracks=dict(frames=frames, xy=xy.astype(np.float32), vis=vis, uv=uv, corner_ids=corner_ids),
                depth=np.array(depths), depth_drift=drift, truth=dict(young=young, air=sim.air, vertices=Xs, hands=Hn, eye=eye, R=Rwc))
