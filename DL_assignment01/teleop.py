#!/usr/bin/env python3
"""Drive the robot's stick from the keyboard in the MuJoCo viewer and save each episode as a demonstration
(sim/episode_io.py format). Push the rope (red keypoints) onto the goal T (green keypoints).

    python teleop.py --student 20261234                 # demos/20261234/ep_<seed>.npz, seeds from your student id
    python teleop.py --student 20261234 --keys viewer   # if the held keys are not read (macOS / Wayland, see README)

Keys (--keys hold, the default; read system-wide with pynput, so keep the viewer window in front):
    arrows          hold: stick tip in the table plane, relative to the view (up = away from the camera)
    PgUp / PgDn     hold: stick tip up / down
    Shift           hold: full speed (160 mm/s xy, 80 mm/s z); otherwise half
Keys (--keys viewer; MuJoCo's viewer only reports presses, so each press sets the motion until the next):
    arrows, PgUp / PgDn   start moving that way (half speed);  End   stop
Both modes (viewer window focused):
    Enter   save this episode and go on to the next seed
    Delete  discard this episode and restart the same seed
Close the viewer window to quit (the current episode is NOT saved). An episode also ends, and is saved, after --steps
steps (10 Hz, 600 = 60 s), or when the arm has been blocked for 1 s; one in which no key was pressed is never saved.
"""
import argparse, sys, threading, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'sim'))
ENTER, DELETE, END = 257, 261, 269                                              # GLFW key codes (viewer key_callback)
VIEWER_DIRS = {265: 'up', 264: 'down', 263: 'left', 262: 'right', 266: 'z+', 267: 'z-'}


class Keys:
    """The keys held now, and the last command key pressed in the viewer window."""
    def __init__(self, mode):
        self.held, self.lock, self.cmd, self.mode = set(), threading.Lock(), None, mode
        if mode == 'hold':
            from pynput import keyboard as kb
            K = kb.Key
            self.names = {K.up: 'up', K.down: 'down', K.left: 'left', K.right: 'right', K.page_up: 'z+', K.page_down: 'z-', K.shift: 'shift', K.shift_r: 'shift'}
            kb.Listener(on_press=lambda k: self._set(k, True), on_release=lambda k: self._set(k, False), daemon=True).start()

    def _set(self, k, down):
        n = self.names.get(k)
        if n is None: return
        with self.lock: (self.held.add if down else self.held.discard)(n)

    def viewer_key(self, code):
        if code == ENTER: self.cmd = 'save'
        elif code == DELETE: self.cmd = 'restart'
        elif self.mode == 'viewer' and code == END:
            with self.lock: self.held.clear()
        elif self.mode == 'viewer' and code in VIEWER_DIRS:
            with self.lock: self.held = {VIEWER_DIRS[code]}

    def action(self, azimuth, R_tb):
        """Held keys -> (normalised base-frame stick-tip velocity, env.step's action; whether any key is held). xy follows
        the view's azimuth."""
        from rope_env import VMAX, VZ
        with self.lock: h = set(self.held)
        f, r = (('up' in h) - ('down' in h)), (('right' in h) - ('left' in h))
        az = np.radians(azimuth); fwd, right = np.array([np.cos(az), np.sin(az)]), np.array([np.sin(az), -np.cos(az)])
        xy = f * fwd + r * right; xy /= max(np.linalg.norm(xy), 1.)
        s = np.array([VMAX, VMAX, VZ]); v = np.r_[xy, ('z+' in h) - ('z-' in h)] * (1. if 'shift' in h else .5) * s     # table frame, m/s
        return np.clip(R_tb.T @ v / s, -1, 1), bool(h - {'shift'})     # as env.step inverts it: v_tab = R_tb @ (a * s)


BRANCHES = (range(0, 5), range(5, 10), range(10, 19))     # arm, arm, stem; 0, 5 and 10 are all the junction
RED, GREEN = np.array([.9, .1, .1, 1], np.float32), np.array([.2, .8, .3, .6], np.float32)


def draw_kp(scn, p, rgba, r):
    """Keypoints p (19, 3) into the viewer's user scene: spheres of radius r joined along each branch."""
    import mujoco
    S, C = mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE
    for br in BRANCHES:
        for i in br:
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], S, np.r_[r, 0, 0], p[i], np.eye(3).ravel(), rgba); scn.ngeom += 1
        for a, b in zip(br[:-1], br[1:]):
            g = scn.geoms[scn.ngeom]; mujoco.mjv_initGeom(g, C, np.zeros(3), np.zeros(3), np.eye(3).ravel(), rgba)
            mujoco.mjv_connector(g, C, r / 2, p[a], p[b]); scn.ngeom += 1


def draw(v, env):
    """The goal keypoints on the table, the rope's on its top (its axis is 1 cm up, inside the rope)."""
    from rope_env import RADIUS
    scn = v.user_scn; scn.ngeom = 0
    draw_kp(scn, np.c_[env.goal, np.full(len(env.goal), .002)], GREEN, .005)
    draw_kp(scn, np.c_[env.kp, env.kp_z + RADIUS + .002], RED, .004)


def episode(seed, args, keys, cam):
    """One teleoperated episode. Returns 'save' / 'restart' / 'quit' (what ended it); saves on 'save'."""
    import mujoco, mujoco.viewer
    from rope_env import CENTRE
    from scene import make_env
    from episode_io import Recorder
    env, obs = make_env(seed, args.mu, args.steps); mu = env.p['mu_table']
    rec = Recorder(env, seed, mu, args.student, 'teleop')
    print(f'\nseed {seed}: friction {mu:.3f}. Enter saves, Delete restarts, closing the window quits.', flush=True)
    keys.cmd, end, info = None, 'save', {}
    with mujoco.viewer.launch_passive(env.m, env.d, key_callback=keys.viewer_key) as v:
        with v.lock():
            if cam: v.cam.lookat[:], v.cam.distance, v.cam.azimuth, v.cam.elevation = cam
            else: v.cam.lookat[:], v.cam.distance, v.cam.azimuth, v.cam.elevation = np.r_[CENTRE, 0], .9, 90., -55.
            draw(v, env)
        v.sync(); t_next = time.perf_counter()
        for k in range(args.steps):
            if not v.is_running(): end = 'quit'; break
            if keys.cmd: end = keys.cmd; break
            with v.lock():
                a, human = keys.action(v.cam.azimuth, env.R_tb)
                rec.before(obs); obs, _, done, info = env.step(a); rec.after(info, human)
                draw(v, env)
            v.sync()
            if k % 5 == 0:
                e, tip = info['err'], env.target * 1000
                print(f'\r  step {k:4d}  tip {tip[0]:6.0f} {tip[1]:6.0f} {tip[2]:4.0f} mm   pos {e["pos"]:5.1f} mm  rot {e["rot"]:5.1f} deg  '
                      f'shape {e["shape"]:5.1f} mm  {"SUCCESS" if info["success"] else "       "}{"  blocked" if info["blocked"] else "        "}', end='', flush=True)
            if done: break
            t_next += .1; time.sleep(max(t_next - time.perf_counter(), 0))     # 10 Hz, as the real robot
        cam[:] = [v.cam.lookat.copy(), v.cam.distance, v.cam.azimuth, v.cam.elevation]
    print()
    if end != 'save' or len(rec) == 0: print('  not saved'); return end
    if not any(rec.rows['human']): print('  no key was pressed: not saved, the same seed again'); return 'restart'
    f = rec.save(args.out / f'ep_{seed}.npz'); ok = bool(info.get('success', False))
    print(f'  {len(rec)} steps ({sum(rec.rows["human"])} driven), success {ok} -> {f}'); return end


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--student', required=True, help='your student id (sets your seeds and the output folder)')
    ap.add_argument('--keys', choices=['hold', 'viewer'], default='hold', help='hold: pynput, held keys (default); viewer: MuJoCo key presses only')
    ap.add_argument('--mu', type=float, help='table friction (default: the fitted 0.576)')
    ap.add_argument('--steps', type=int, default=600, help='episode length at 10 Hz (600 = 60 s)')
    ap.add_argument('--out', type=Path, help='output folder (default: demos/<student>)')
    args = ap.parse_args()
    from scene import student_seeds
    seeds = student_seeds(args.student); args.out = args.out or ROOT / 'demos' / args.student; args.out.mkdir(parents=True, exist_ok=True)
    done = {int(f.stem.split('_')[1]) for f in args.out.glob('ep_*.npz')}
    todo = [s for s in seeds if s not in done]
    print(f'student {args.student}: seeds {seeds.start}-{seeds.stop - 1}, {len(done)} saved already in {args.out}')
    keys, cam = Keys(args.keys), []
    while todo:
        end = episode(todo[0], args, keys, cam)
        if end == 'quit': break
        if end == 'save': todo.pop(0)


if __name__ == '__main__':
    main()
