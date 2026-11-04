#!/usr/bin/env python3
"""Milestone 0: check the install. Builds the scene for one seed, runs 50 steps with a fixed action (no window), checks
the observation, saves a picture of the scene, and writes a report.

    python scripts/check_env.py --student 20261234
Writes runs/m0/check_env.json and runs/m0/scene.png (deliverables of Milestone 0).
"""
import argparse, json, platform, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sim'))


def render_png(env, path):
    """The scene from above the workspace (offscreen OpenGL) with teleop.py's keypoints (red rope, green goal), or the
    reason it failed."""
    try:
        import mujoco, cv2
        sys.path.insert(0, str(ROOT)); from teleop import draw_kp, RED, GREEN
        from rope_env import CENTRE, RADIUS
        r = mujoco.Renderer(env.m, 480, 640); cam = mujoco.MjvCamera()
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = np.r_[CENTRE, 0], .9, 90., -55.
        r.update_scene(env.d, camera=cam)
        draw_kp(r.scene, np.c_[env.goal, np.full(len(env.goal), .002)], GREEN, .005); draw_kp(r.scene, np.c_[env.kp, env.kp_z + RADIUS + .002], RED, .004)
        ok, buf = cv2.imencode('.png', cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)); r.close()
        path.write_bytes(buf.tobytes()); return 'ok' if ok and path.stat().st_size > 0 else 'failed: empty image'     # not cv2.imwrite: it fails on non-ASCII paths (Windows)
    except Exception as e: return f'failed: {e!r}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--student', required=True); ap.add_argument('--out', type=Path, default=ROOT / 'runs' / 'm0')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    import mujoco, robosuite
    from scene import make_env, student_seeds, OBS_DIM, OBS_LAYOUT
    seed = student_seeds(a.student).start
    t0 = time.perf_counter(); env, obs = make_env(seed); t_build = time.perf_counter() - t0
    checks = {'obs_shape': list(obs.shape) == [OBS_DIM], 'obs_finite': bool(np.isfinite(obs).all())}
    tip0 = env.tip_of(env.q).copy(); t0 = time.perf_counter()
    for _ in range(50): obs, r, done, info = env.step([0., 0., -.5])          # lower the stick for 5 s
    t_step = (time.perf_counter() - t0) / 50
    tip1 = env.tip_of(env.q)
    checks['stick_moved_down'] = bool(tip1[2] < tip0[2] - .005) or bool(tip1[2] < .005)
    checks['realtime_10hz'] = t_step < .1
    checks['still_finite'] = bool(np.isfinite(obs).all()) and not info['bad']
    png = render_png(env, a.out / 'scene.png')
    rep = dict(student=a.student, seed=seed, time=time.strftime('%Y-%m-%d %H:%M:%S'), python=platform.python_version(), platform=platform.platform(),
               mujoco=mujoco.__version__, robosuite=robosuite.__version__, numpy=np.__version__, build_s=round(t_build, 2), step_ms=round(t_step * 1000, 1),
               tip_start_mm=(tip0 * 1000).round(1).tolist(), tip_end_mm=(tip1 * 1000).round(1).tolist(), err_after_50=info['err'],
               obs_layout={k: [s.start, s.stop] for k, s in OBS_LAYOUT.items()}, render=png, checks=checks, passed=all(checks.values()))
    (a.out / 'check_env.json').write_text(json.dumps(rep, indent=2))
    for k, v in checks.items(): print(f'  [{"PASS" if v else "FAIL"}] {k}')
    print(f'  render: {png}\n  build {t_build:.1f} s, step {t_step * 1000:.1f} ms (real time needs < 100 ms)')
    print(f'{"PASSED" if rep["passed"] else "FAILED"} -> {a.out / "check_env.json"}')
    sys.exit(0 if rep['passed'] else 1)


if __name__ == '__main__':
    main()
