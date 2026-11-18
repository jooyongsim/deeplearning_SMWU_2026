#!/usr/bin/env python3
"""Fold the cloth yourself: drive the gripper from the keyboard in the MuJoCo viewer (Milestone 2). Each episode is
saved in sim/episode_io.py's format (demos/<team>/ep_<seed>.npz) and can be turned into a video.

    python teleop.py --student T03                              # the example cloth
    python teleop.py --student T03 --mesh assets/my_cloth       # your own cloth (Milestone 1)
    python teleop.py --student T03 --keys viewer                # if held keys are not read (macOS / Wayland, see README)

Keys (--keys hold, the default; read with pynput, keep the viewer window in front):
    arrows          hold: gripper in the table plane, relative to the view (up = away from the camera)
    PgUp / PgDn     hold: gripper up / down
    Shift           hold: full speed (250 mm/s); otherwise half
    Home            close / open the gripper (closing grasps the cloth within 3 cm of the fingertips)
Keys (--keys viewer; the viewer only reports presses, so each press sets the motion until the next):
    arrows, PgUp / PgDn   start moving that way (half speed);  End  stop;  Home  close / open
Both modes (viewer window focused):
    Enter   save this episode and go on to the next seed
    Delete  discard this episode and restart the same seed
Close the viewer window to quit (the current episode is NOT saved). Do not press Space (the viewer pauses) or
Backspace (the viewer resets the simulation).
The terminal shows the corner gaps (c0-c2, c1-c3), the covered area and SUCCESS (both gaps <= 70 mm, area <= 0.35).
"""
import argparse, sys, threading, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'sim'))
ENTER, DELETE, END, HOME = 257, 261, 269, 268                                   # GLFW key codes (viewer key_callback)
VIEWER_DIRS = {265: 'up', 264: 'down', 263: 'left', 262: 'right', 266: 'z+', 267: 'z-'}
CORNER_RGBA = [np.array(c, np.float32) for c in ([.9, .1, .1, 1], [.1, .3, .9, 1], [.9, .1, .1, 1], [.1, .3, .9, 1])]   # pairs: red c0-c2, blue c1-c3


class Keys:
    def __init__(self, mode):
        self.held, self.lock, self.cmd, self.mode, self.grip = set(), threading.Lock(), None, mode, False
        if mode == 'hold':
            from pynput import keyboard as kb
            K = kb.Key
            self.names = {K.up: 'up', K.down: 'down', K.left: 'left', K.right: 'right', K.page_up: 'z+', K.page_down: 'z-', K.shift: 'shift', K.shift_r: 'shift'}
            def press(k):
                if k == K.home: self.grip = not self.grip
                else: self._set(k, True)
            kb.Listener(on_press=press, on_release=lambda k: self._set(k, False), daemon=True).start()

    def _set(self, k, down):
        n = self.names.get(k)
        if n is None: return
        with self.lock: (self.held.add if down else self.held.discard)(n)

    def viewer_key(self, code):
        if code == ENTER: self.cmd = 'save'
        elif code == DELETE: self.cmd = 'restart'
        elif self.mode == 'viewer' and code == HOME: self.grip = not self.grip
        elif self.mode == 'viewer' and code == END:
            with self.lock: self.held.clear()
        elif self.mode == 'viewer' and code in VIEWER_DIRS:
            with self.lock: self.held = {VIEWER_DIRS[code]}

    def action(self, azimuth):
        """Held keys -> (env.step's action [vx, vy, vz, grip] in the table frame, whether a key is held)."""
        with self.lock: h = set(self.held)
        f, r = (('up' in h) - ('down' in h)), (('right' in h) - ('left' in h))
        az = np.radians(azimuth); fwd, right = np.array([np.cos(az), np.sin(az)]), np.array([np.sin(az), -np.cos(az)])
        xy = f * fwd + r * right; xy /= max(np.linalg.norm(xy), 1.)
        v = np.r_[xy, ('z+' in h) - ('z-' in h)] * (1. if 'shift' in h else .5)
        return np.r_[v, 1. if self.grip else -1.], bool(h - {'shift'})


def draw(v, env):
    """Corner markers (red pair c0-c2, blue pair c1-c3) and the grasp radius under the gripper."""
    import mujoco
    from cloth_env import GRASP_R
    scn = v.user_scn; scn.ngeom = 0; X = env.verts()
    for c, rgba in zip(env.mesh['corners'], CORNER_RGBA):
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.r_[.007, 0, 0], X[c] + [0, 0, .004], np.eye(3).ravel(), rgba); scn.ngeom += 1
    p = env.tcp(); col = np.array([.1, .8, .2, .35] if env.closed else [1, .8, .1, .35], np.float32)
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER, np.r_[GRASP_R, .0015, 0], np.r_[p[:2], .001], np.eye(3).ravel(), col); scn.ngeom += 1


def episode(seed, args, keys, cam):
    import mujoco, mujoco.viewer
    from cloth_env import CENTRE
    from scene import make_env
    from episode_io import Recorder
    env, obs = make_env(seed, args.mesh, args.steps); rec = Recorder(env, seed, args.mesh or 'cloth_example', args.student, 'teleop')
    print(f'\nseed {seed}: Home closes / opens the gripper, Enter saves, Delete restarts, closing the window quits.', flush=True)
    keys.cmd, keys.grip, end, info = None, False, 'save', env.info
    with mujoco.viewer.launch_passive(env.m, env.d, key_callback=keys.viewer_key) as v:
        with v.lock():
            if cam: v.cam.lookat[:], v.cam.distance, v.cam.azimuth, v.cam.elevation = cam
            else: v.cam.lookat[:], v.cam.distance, v.cam.azimuth, v.cam.elevation = np.r_[CENTRE, 0], .95, 90., -55.
            draw(v, env)
        v.sync(); t_next = time.perf_counter()
        for k in range(args.steps):
            if not v.is_running(): end = 'quit'; break
            if keys.cmd: end = keys.cmd; break
            with v.lock():
                a, human = keys.action(v.cam.azimuth)
                rec.before(obs, human or (a[3] > 0) != env.closed); obs, _, done, info = env.step(a); rec.after(info)
                draw(v, env)
            v.sync()
            if k % 5 == 0:
                p = env.tcp() * 1000
                print(f'\r  t {k / 10:5.1f} s  gripper {p[0]:5.0f} {p[1]:5.0f} {p[2]:4.0f} mm  {"CLOSED" if env.closed else "open  "} held {info["held"]}  '
                      f'gaps {info["gap_a"]:4.0f} / {info["gap_b"]:4.0f} mm  cover {info["cover"]:.2f}  {"SUCCESS" if info["success"] else "       "}', end='', flush=True)
            if done: break
            t_next += .1; time.sleep(max(t_next - time.perf_counter(), 0))
        cam[:] = [v.cam.lookat.copy(), v.cam.distance, v.cam.azimuth, v.cam.elevation]
    print()
    if end != 'save' or len(rec) == 0: print('  not saved'); return end
    if not any(rec.rows['human']): print('  no key was pressed: not saved, the same seed again'); return 'restart'
    f = rec.save(args.out / f'ep_{seed}.npz')
    print(f'  {len(rec)} steps, gaps {info["gap_a"]:.0f} / {info["gap_b"]:.0f} mm, cover {info["cover"]:.2f}, success {info["success"]} -> {f}'); return end


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--student', required=True, help='your team id, T01-T05')
    ap.add_argument('--mesh', default=None, help='cloth mesh folder (default: assets/cloth_example)')
    ap.add_argument('--keys', choices=['hold', 'viewer'], default='hold')
    ap.add_argument('--steps', type=int, default=1200, help='episode length at 10 Hz (1200 = 2 min)')
    ap.add_argument('--out', type=Path, help='output folder (default: demos/<team>)')
    args = ap.parse_args()
    from scene import student_seeds
    seeds = student_seeds(args.student); args.out = args.out or ROOT / 'demos' / args.student; args.out.mkdir(parents=True, exist_ok=True)
    done = {int(f.stem.split('_')[1]) for f in args.out.glob('ep_*.npz')}
    todo = [s for s in seeds if s not in done]
    print(f'team {args.student}: seeds {seeds.start}-{seeds.stop - 1}, {len(done)} saved already in {args.out}')
    keys, cam = Keys(args.keys), []
    while todo:
        end = episode(todo[0], args, keys, cam)
        if end == 'quit': break
        if end == 'save': todo.pop(0)


if __name__ == '__main__':
    main()
