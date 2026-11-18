#!/usr/bin/env python3
"""Render saved episodes (teleop demos, evaluation rollouts) to mp4 (H.264: plays in PowerPoint and on YouTube), or to
a contact sheet of frames (--sheet). Uses the recorded MuJoCo states (qpos): nothing is re-simulated, the video is
exactly what was recorded. Text: time, corner gaps, coverage, SUCCESS.

    python scripts/render_video.py runs/m4/ppo/rollouts/ep_900000.npz            # -> runs/m4/ppo/videos/ep_900000.mp4
    python scripts/render_video.py runs/m4/ppo/rollouts/ep_9000*.npz --out runs/m4/ppo/videos
    python scripts/render_video.py demos/T03/ep_300.npz --speed 2                 # a teleop demo, twice as fast
    python scripts/render_video.py demos/T03/ep_300.npz --sheet                   # one png with 12 frames
"""
import argparse, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sim')); sys.path.insert(0, str(ROOT))
W, H = 960, 720       # multiples of 16 (H.264 macro blocks)


def env_for(ep):
    from scene import make_env, DEFAULT_MESH
    mesh = ep.get('mesh', 'cloth_example'); mesh = DEFAULT_MESH if mesh in ('', 'None', 'cloth_example') else mesh
    env, _ = make_env(int(ep['seed']), mesh); return env


def see_through(m, alpha=.3):
    """The arm and gripper see-through, so the cloth under the gripper stays visible."""
    import mujoco
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ''
        if name.startswith(('robot0_', 'gripper0_', 'mount0_')) and m.geom_rgba[g, 3] > 0:
            m.geom_rgba[g, 3] = alpha
            if m.geom_matid[g] >= 0: m.mat_rgba[m.geom_matid[g], 3] = alpha


def camera(view='persp'):
    import mujoco
    from cloth_env import CENTRE
    cam = mujoco.MjvCamera(); cam.lookat[:] = np.r_[CENTRE + [0, .02], 0]
    if view == 'top': cam.distance, cam.azimuth, cam.elevation = .95, 90., -89.
    else: cam.distance, cam.azimuth, cam.elevation = .95, 90., -58.
    return cam


def frames(ep, every=1, view='persp', w=W, h=H, alpha=.3):
    """Yields (k, rgb image with the text) for the recorded steps."""
    import cv2, mujoco
    env = env_for(ep); m = env.m; see_through(m, alpha)
    r = mujoco.Renderer(m, h, w); cam = camera(view)
    T = int(ep['length']); ks = list(range(0, T, every)) + ([T - 1] if (T - 1) % every else [])
    try:
        for k in ks:
            env.d.qpos[:] = ep['qpos'][k]; mujoco.mj_forward(m, env.d); r.update_scene(env.d, camera=cam)
            img = np.ascontiguousarray(r.render()); g = ep['gaps'][k]; ok = bool(ep['succ'][k]); s = h / 720
            lines = [f'{ep["source"]}  seed {int(ep["seed"])}  t {k / 10:5.1f} s', f'corner gaps {g[0]:4.0f} / {g[1]:4.0f} mm   cover {ep["cover"][k]:.2f}']
            for i, txt in enumerate(lines):
                for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
                    cv2.putText(img, txt, (int(14 * s), int((32 + 30 * i) * s)), cv2.FONT_HERSHEY_SIMPLEX, .75 * s, col, max(1, int(th * s)), cv2.LINE_AA)
            if ok: cv2.putText(img, 'SUCCESS', (int(14 * s), int(124 * s)), cv2.FONT_HERSHEY_SIMPLEX, 1. * s, (40, 200, 40), max(1, int(3 * s)), cv2.LINE_AA)
            yield k, img
    finally: r.close()


def render(ep, path, speed=1., title=None, view='persp'):
    import imageio_ffmpeg
    out = imageio_ffmpeg.write_frames(str(path), (W, H), fps=10 * speed, codec='libx264', pix_fmt_out='yuv420p', quality=7, macro_block_size=16)
    out.send(None)
    for _, img in frames(ep, view=view): out.send(img.tobytes())
    out.close()


def sheet(ep, path, n=12, cols=4, view='persp'):
    import cv2
    T = int(ep['length']); every = max(1, T // (n - 1))
    imgs = [img for _, img in frames(ep, every, view, 480, 360)][:n]
    imgs += [np.zeros_like(imgs[0])] * (-len(imgs) % cols)
    grid = np.vstack([np.hstack(imgs[i:i + cols]) for i in range(0, len(imgs), cols)])
    ok, buf = cv2.imencode('.png', cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)); Path(path).write_bytes(buf.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('episodes', nargs='+', type=Path); ap.add_argument('--out', type=Path)
    ap.add_argument('--speed', type=float, default=1., help='playback speed (1 = real time, 10 Hz)')
    ap.add_argument('--view', choices=['persp', 'top'], default='persp')
    ap.add_argument('--sheet', action='store_true', help='a png of 12 frames instead of an mp4')
    a = ap.parse_args()
    from episode_io import load, problems
    for f in a.episodes:
        ep = load(f); bad = problems(ep)
        if bad: print(f'{f}: skipped, {"; ".join(bad)}'); continue
        out = a.out or (f.parent.parent / 'videos' if f.parent.name == 'rollouts' else f.parent / 'videos'); out.mkdir(parents=True, exist_ok=True)
        if a.sheet: p = out / f'{f.stem}.png'; sheet(ep, p, view=a.view)
        else: p = out / f'{f.stem}.mp4'; render(ep, p, a.speed, view=a.view)
        print(f'{f.name}: {int(ep["length"])} steps, success {bool(ep["success"])} -> {p}', flush=True)


if __name__ == '__main__':
    main()
