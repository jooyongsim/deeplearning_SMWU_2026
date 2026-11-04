#!/usr/bin/env python3
"""Render saved episodes (teleop demos or eval rollouts) to mp4 (H.264, plays in PowerPoint and on YouTube). Uses the
recorded MuJoCo states (qpos), so nothing is re-simulated: the video shows exactly what was recorded. Red: the rope's
keypoints, green: the goal T; the text shows the step, the errors and SUCCESS.

    python scripts/render_video.py runs/m4/bc/rollouts/ep_900000.npz                     # -> runs/m4/bc/videos/ep_900000.mp4
    python scripts/render_video.py runs/m4/bc/rollouts/ep_9000*.npz --out runs/m4/bc/videos
    python scripts/render_video.py demos/20261234/ep_123400.npz --out runs/m1 --speed 2   # a demo, twice as fast
Default output: a videos/ folder next to the episode's folder (rollouts/ -> videos/).
"""
import argparse, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sim')); sys.path.insert(0, str(ROOT))
W, H = 960, 720       # multiples of 16 (H.264 macro blocks)


def render(ep, path, speed=1., title=''):
    import cv2, mujoco, imageio_ffmpeg
    from scene import make_env
    from rope_env import CENTRE, RADIUS
    from teleop import draw_kp, RED, GREEN
    env, _ = make_env(int(ep['seed']), float(ep['mu']))
    m = env.m
    for g in range(m.ngeom):      # the arm and gripper see-through (the rope stays visible under them); the stick opaque
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ''
        if name.startswith(('robot0_', 'gripper0_', 'mount0_')):
            m.geom_rgba[g, 3] = .25
            if m.geom_matid[g] >= 0: m.mat_rgba[m.geom_matid[g], 3] = .25
    r = mujoco.Renderer(m, H, W); cam = mujoco.MjvCamera()
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = np.r_[CENTRE, 0], .8, 90., -60.
    out = imageio_ffmpeg.write_frames(str(path), (W, H), fps=10 * speed, codec='libx264', pix_fmt_out='yuv420p', quality=7, macro_block_size=16)
    out.send(None)
    goal = np.c_[ep['goal'], np.full(len(ep['goal']), .002)]
    for k, q in enumerate(ep['qpos']):
        env.d.qpos[:] = q; mujoco.mj_forward(env.m, env.d); r.update_scene(env.d, camera=cam)
        draw_kp(r.scene, goal, GREEN, .005); draw_kp(r.scene, np.c_[ep['kp'][k], ep['kp_z'][k] + RADIUS + .002], RED, .004)
        img = np.ascontiguousarray(r.render()); e = ep['err'][k]; ok = bool(ep['succ'][k])
        lines = [f'{title}  seed {int(ep["seed"])}  t {k / 10:5.1f} s', f'pos {e[0]:5.1f} mm  rot {e[1]:5.1f} deg  shape {e[2]:5.1f} mm']
        for i, s in enumerate(lines):
            for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)): cv2.putText(img, s, (14, 32 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, .75, col, th, cv2.LINE_AA)
        if ok: cv2.putText(img, 'SUCCESS', (14, 32 + 60), cv2.FONT_HERSHEY_SIMPLEX, 1., (40, 200, 40), 3, cv2.LINE_AA)
        out.send(img.tobytes())
    out.close(); r.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('episodes', nargs='+', type=Path); ap.add_argument('--out', type=Path)
    ap.add_argument('--speed', type=float, default=1., help='playback speed (1 = real time, 10 Hz)')
    a = ap.parse_args()
    from episode_io import load, problems
    for f in a.episodes:
        ep = load(f); bad = problems(ep)
        if bad: print(f'{f}: skipped, {"; ".join(bad)}'); continue
        out = a.out or (f.parent.parent / 'videos' if f.parent.name == 'rollouts' else f.parent / 'videos'); out.mkdir(parents=True, exist_ok=True)
        p = out / f'{f.stem}.mp4'; render(ep, p, a.speed, str(ep['source']))
        print(f'{f.name}: {int(ep["length"])} steps, success {bool(ep["success"])} -> {p}', flush=True)


if __name__ == '__main__':
    main()
