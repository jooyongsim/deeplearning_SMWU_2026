#!/usr/bin/env python3
"""Milestone 0: check the install. Builds the scene with the cloth mesh, lowers the gripper onto the cloth, grasps a
corner, lifts it 10 cm and lets go (no window), checks the cloth followed and the simulation stayed sane, saves a
picture, and writes a report.

    python scripts/check_env.py --student T03
    python scripts/check_env.py --student T03 --mesh assets/my_cloth      # after Milestone 1: your own cloth
Writes runs/m0/check_env.json and runs/m0/scene.png.
"""
import argparse, json, platform, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sim'))


def render_png(env, path, w=800, h=600):
    try:
        import mujoco, cv2
        from cloth_env import CENTRE
        r = mujoco.Renderer(env.m, h, w); cam = mujoco.MjvCamera()
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = np.r_[CENTRE, .03], .95, 90., -50.
        r.update_scene(env.d, camera=cam); img = r.render(); r.close()
        ok, buf = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR)); path.write_bytes(buf.tobytes())   # not cv2.imwrite: non-ASCII paths
        return 'ok' if ok else 'failed: encode'
    except Exception as e: return f'failed: {e!r}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--student', required=True); ap.add_argument('--mesh', default=None)
    ap.add_argument('--out', type=Path, default=ROOT / 'runs' / 'm0')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    import mujoco, robosuite
    from scene import make_env, student_seeds, OBS_DIM, OBS_LAYOUT
    seed = student_seeds(a.student).start
    t0 = time.perf_counter(); env, obs = make_env(seed, a.mesh); t_build = time.perf_counter() - t0
    checks = {'obs_shape': list(obs.shape) == [OBS_DIM], 'obs_finite': bool(np.isfinite(obs).all()),
              'mujoco_3.3.1': mujoco.__version__ == '3.3.1'}
    X0 = env.verts(); c0 = env.mesh['corners'][0]; corner = X0[c0]; t0 = time.perf_counter(); n = 0
    def go(target, grip, steps):
        nonlocal n; info = env.info
        for _ in range(steps):
            v = (np.asarray(target) - env.tcp()) / (.25 * .1); v = np.clip(v, -1, 1); _, _, _, info = env.step(np.r_[v, grip]); n += 1
        return info
    go(np.r_[corner[:2], .05], -1, 12); go(np.r_[corner[:2], corner[2] + .002], -1, 6)
    info = go(env.tcp(), 1, 1); held = info['held']
    info = go(np.r_[corner[:2], .10], 1, 6); lifted = env.verts()[c0, 2]
    t_step = (time.perf_counter() - t0) / n
    go(env.tcp(), -1, 10)
    checks['grasped_corner'] = held > 0
    checks['corner_lifted'] = bool(lifted > .07)
    checks['rest_of_cloth_on_table'] = bool(np.median(env.verts()[:, 2]) < .01)
    checks['still_finite'] = bool(np.isfinite(env.verts()).all()) and not env.info['bad']
    checks['speed_ok'] = t_step < .5
    png = render_png(env, a.out / 'scene.png')
    rep = dict(student=a.student, seed=seed, mesh=a.mesh or 'assets/cloth_example', time=time.strftime('%Y-%m-%d %H:%M:%S'), python=platform.python_version(),
               platform=platform.platform(), mujoco=mujoco.__version__, robosuite=robosuite.__version__, numpy=np.__version__,
               cloth_vertices=env.N, cloth_cm=[round(env.mesh['width'] * 100, 1), round(env.mesh['height'] * 100, 1)],
               build_s=round(t_build, 2), step_ms=round(t_step * 1000, 1), held_vertices=int(held), corner_lift_mm=round(float(lifted) * 1000, 1),
               obs_layout={k: [s.start, s.stop] for k, s in OBS_LAYOUT.items()}, render=png, checks=checks, passed=all(checks.values()))
    (a.out / 'check_env.json').write_text(json.dumps(rep, indent=2))
    for k, v in checks.items(): print(f'  [{"PASS" if v else "FAIL"}] {k}')
    print(f'  cloth {rep["cloth_cm"][0]} x {rep["cloth_cm"][1]} cm, {env.N} vertices; grasped {held}, corner lifted to {lifted * 1000:.0f} mm')
    print(f'  render: {png}\n  build {t_build:.1f} s, step {t_step * 1000:.0f} ms (0.1 s of robot motion; real time needs < 100 ms)')
    print(f'{"PASSED" if rep["passed"] else "FAILED"} -> {a.out / "check_env.json"}')
    sys.exit(0 if rep['passed'] else 1)


if __name__ == '__main__':
    main()
