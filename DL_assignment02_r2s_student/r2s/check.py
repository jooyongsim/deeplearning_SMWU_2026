"""Tests for your TODO functions (r2s/lift.py: backproject, per_frame_scale, floor_frame; r2s/fit.py: frame_loss,
episode_loss, curve_min, upper_bound) on small made-up inputs with known answers.

    import check; check.run()                 # in the notebook, after your TODO cells (which replace the stubs)
    python r2s/check.py                       # locally
"""
import json, sys, time, traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'sim'), str(ROOT / 'r2s')]


def run(verbose=True, out=None):
    import lift, fit
    res = {}; rng = np.random.default_rng(1)

    def t(name):
        def deco(f):
            try: f(); res[name] = 'PASS'
            except NotImplementedError as e: res[name] = f'FAIL: not implemented ({e})'
            except Exception as e: res[name] = f'FAIL: {e!r}'
        return deco

    K = np.array([[600., 0, 480], [0, 600, 270], [0, 0, 1]])

    @t('backproject: the principal point at depth 2 m is (0, 0, 2)')
    def _():
        p = lift.backproject(np.array([480., 270.]), np.array(2.), K); assert np.allclose(p, [0, 0, 2]), p

    @t('backproject: shapes (..., 2) and (...) -> (..., 3), an off-axis point')
    def _():
        p = lift.backproject(np.array([[[1080., 870.]]]), np.array([[1.5]]), K); assert p.shape == (1, 1, 3), p.shape
        assert np.allclose(p[0, 0], [1.5, 1.5, 1.5]), p

    uv = np.array([(u, v) for v in np.linspace(0, 1, 5) for u in np.linspace(0, 1, 5)]); w, h = .278, .32
    pairs, L = lift.neighbour_pairs(uv, w, h)
    flat = np.c_[uv[:, 0] * w, uv[:, 1] * h, np.zeros(len(uv))]

    @t('per_frame_scale: a frame measured 1.5x too large gets 1/1.5, another 0.8x gets 1/0.8')
    def _():
        X = np.stack([flat * 1.5 + [0, 0, 1], flat * .8 + [0, 0, 1]]); s = lift.per_frame_scale(X, np.ones((2, len(uv)), bool), pairs, L)
        assert np.allclose(s, [1 / 1.5, 1 / .8], atol=1e-6), s

    @t('per_frame_scale: hidden points are not used; a frame with too few visible pairs gives NaN')
    def _():
        X = np.stack([flat * 2 + [0, 0, 1]] * 2); X[0, :5] += 5.                      # garbage on 5 points...
        vis = np.ones((2, len(uv)), bool); vis[0, :5] = False; vis[1, :] = False; vis[1, :3] = True   # ...that are hidden
        s = lift.per_frame_scale(X, vis, pairs, L); assert abs(s[0] - .5) < 1e-6 and np.isnan(s[1]), s

    @t('floor_frame: a floor tilted 20 deg in the camera frame becomes z = 0, z up, a proper rotation')
    def _():
        a = np.radians(20); n_cam = np.array([0, -np.cos(a), -np.sin(a)])
        u = np.cross(n_cam, [1, 0, 0]); u /= np.linalg.norm(u); v = np.cross(n_cam, u)
        P = np.array([0, 1.2, 2.]) + rng.uniform(-1, 1, (3000, 1)) * u + rng.uniform(-1, 1, (3000, 1)) * v + rng.normal(0, .002, (3000, 1)) * n_cam
        R, tw, frac = lift.floor_frame(P); Q = P @ R.T + tw
        assert np.abs(Q[:, 2]).mean() < .01, np.abs(Q[:, 2]).mean(); assert abs(np.linalg.det(R) - 1) < 1e-6
        above = (np.array([0, 1.2, 2.]) + .5 * n_cam) @ R.T + tw; assert above[2] > .4, above

    S = rng.uniform(-.1, .1, (60, 3))

    @t('frame_loss: same points -> 0; 1 cm off -> 1 cm; works for 2-D (pixels) too')
    def _():
        assert fit.frame_loss(S, S) < 1e-12; assert abs(fit.frame_loss(S + [0, 0, .01], S) - .01) < 1e-9
        assert abs(fit.frame_loss(S[:, :2] + [3., 4.], S[:, :2]) - 5.) < 1e-9

    @t('frame_loss: NaN rows (not seen) are ignored; all NaN -> NaN')
    def _():
        o = S.copy(); o[::2] = np.nan; assert fit.frame_loss(o, S) < 1e-12; assert np.isnan(fit.frame_loss(np.full_like(S, np.nan), S))

    @t('episode_loss: only selected frames count, the first SKIP_S seconds are skipped')
    def _():
        T = 60; tt = np.arange(T) / 30; Sim = np.repeat(S[None], T, 0); Obs = Sim.copy(); Obs[:10] += 1.
        assert fit.episode_loss(Obs, Sim, tt, np.ones(T, bool)) < 1e-12
        Obs[40:] += .02; sel = np.zeros(T, bool); sel[20:40] = True; assert fit.episode_loss(Obs, Sim, tt, sel) < 1e-12

    @t('curve_min: the vertex of a parabola between grid points; a minimum at the grid end gives that end')
    def _():
        x = np.log10([.1, .2, .4, .7, 1.2, 2., 3.5, 6.]); y = 3 * (x - np.log10(.85)) ** 2 + 20
        assert abs(fit.curve_min(x, y) - np.log10(.85)) < 1e-6, fit.curve_min(x, y)
        assert fit.curve_min(x, 5 - x) == x[-1] and fit.curve_min(x, 5 + x) == x[0]

    @t('curve_min: noisy U curve -> within 0.1 decade; stays between the neighbours of the lowest point')
    def _():
        x = np.log10([.1, .2, .4, .7, 1.2, 2., 3.5, 6.]); y = 8 * (x + .1) ** 2 + 22 + np.array([.3, -.2, .1, -.3, .2, 0, -.1, .2])
        b = fit.curve_min(x, y); assert abs(b + .1) < .1, b; i = int(np.argmin(y)); assert x[i - 1] <= b <= x[i + 1], b

    @t('upper_bound: flat then rising -> the first point of the final run above floor + tol; never rising -> inf')
    def _():
        x = np.arange(2., 5.6, .5); y = np.array([22.6, 24.0, 23.1, 22.9, 22.8, 26.3, 27.3, 29.])          # floor (3 lowest) 22.77, band up to 24.77
        b = fit.upper_bound(x, y, 2.); assert abs(b - 4.5) < 1e-9, b
        assert fit.upper_bound(x, np.full(8, 30.) + [0, .3, -.2, .1, 0, .2, -.1, .4], 1.5) == np.inf

    @t('upper_bound: a single noisy point above the band in the flat part does not set the bound')
    def _():
        x = np.arange(2., 5.6, .5); y = np.array([20., 23.5, 20.4, 20.2, 20.8, 20.1, 25., 28.])          # 1e2.5 is a noise spike
        b = fit.upper_bound(x, y, 2.); assert abs(b - 5.0) < 1e-9, b

    @t('upper_bound: the floor is the mean of the 3 lowest points (one lucky low point does not lower the bound)')
    def _():
        x = np.arange(2., 5.6, .5); y = np.array([24.0, 24.3, 24.1, 24.5, 21.4, 23.4, 26.3, 25.7])      # 1e4 is a lucky dip
        b = fit.upper_bound(x, y, 1.5); assert abs(b - 5.0) < 1e-9, b                               # floor 22.8 -> band 24.3

    ok = all(v == 'PASS' for v in res.values())
    if verbose:
        for k, v in res.items(): print(f'  [{"PASS" if v == "PASS" else "FAIL"}] {k}' + ('' if v == 'PASS' else f'\n         {v[6:]}'))
        print(f'{"PASSED" if ok else "FAILED"} ({sum(v == "PASS" for v in res.values())}/{len(res)})')
    if out: Path(out).write_text(json.dumps(dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), passed=ok, tests=res), indent=2))
    return ok, res


if __name__ == '__main__':
    ok, _ = run(out=ROOT / 'check.json'); sys.exit(0 if ok else 1)
