"""Step 5: fit the simulator to the observed cloth -- first the air drag, then an upper bound on the bending stiffness.

The simulator (sim/cloth_sim.py) is driven by the observed hands; every tracked cloth point (known uv) is compared with
the simulated cloth at the same uv, frame by frame, in the image (the simulated points projected through the camera).
Frames up to the take's fit_end are used to fit; the frames after it are held out and only scored.

What a hand-held video can and cannot tell (measured on the course takes and on synthetic videos):
  - air drag `air` is well determined: the loss has a clear U shape in log(air) (no air: the simulated cloth keeps
    swinging; too much: it moves like it is in honey), and the minimum does not depend on the young it is measured at.
  - bending stiffness `young` is NOT determined from below: a hanging cloth whose bending length (a few cm) is much
    smaller than the cloth (30+ cm) takes the shape gravity and the hands give it, whatever its stiffness. Only a
    too-stiff cloth looks wrong. So the loss is flat up to some young and rises after it: we measure that UPPER BOUND,
    and choose a value below it.

So step 5 is three sub-steps (all given, using your TODO functions):
  5a  run_air     young fixed at YOUNG_REF, sweep air on AIR_GRID       -> air = curve_min of the U curve
  5b  run_young   air fixed at that value, sweep young on YOUNG_GRID   -> young_upper = upper_bound of the curve,
                  the tolerance from the simulator's own noise (noise_mm: young x (1 + 1e-9) runs), then 2 more
                  young values between the bound and the grid point below it to tighten it
  5c  choose_young   the log-middle of [YOUNG_RANGE[0], young_upper]: the value whose worst-case error (in decades)
                  is smallest if all we know is that the true young lies in that range

Your TODO functions:
  frame_loss      how far the observed points are from the simulated ones in one frame
  episode_loss    one number for a whole segment (which frames count, how they are averaged)
  curve_min       the minimum of a sampled U curve, between the grid points (a parabola through the 5 lowest-around)
  upper_bound     the first x from which a curve stays above its floor + tol
"""
import os, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SKIP_S = 0.5            # the first 0.5 s are not scored (the simulation starts from its own drape, not the real one)
YOUNG_REF = 10 ** 3.5   # young used while the air is fitted (inside the flat region of every take we measured)
AIR_GRID = (0.1, 0.2, 0.4, 0.7, 1.2, 2., 3.5, 6.)        # N s / m^3 per m^2 of cloth, ~ log-spaced (0.25 decade)
YOUNG_GRID = tuple(10 ** np.arange(2., 5.51, .5))        # 1e2 ... 3e5 (8 values, 0.5 decade)
YOUNG_MIN = 1e2         # the softest young we consider (bending length 0.4 cm: a limp cloth)
TOL_MIN_MM = 2.0        # the smallest tolerance used in upper_bound (mm in the image at the cloth's distance)
FLOOR_K = 3             # the loss floor = mean of the FLOOR_K lowest points (one lucky low point must not set it)
WORKERS = max(1, min(4, os.cpu_count() or 1))           # simulations run in parallel (each is single-threaded)


# --------------------------------------------------------------------------------------------- TODO functions
def frame_loss(obs, sim):
    """obs (P, d) observed cloth points (NaN rows = not seen), sim (P, d) the simulated points at the same uv
    -> mean distance over the seen points (NaN if none is seen). d = 3 (metres, 3-D) or 2 (pixels, in the image)."""
    # TODO: write this in the notebook (it replaces this stub)
    raise NotImplementedError('frame_loss')


def episode_loss(obs, sim, t, sel):
    """obs, sim (T, P, d), t (T,) s, sel (T,) bool: the mean of frame_loss over the selected frames, skipping the
    first SKIP_S seconds of the take (NaN frames ignored)."""
    # TODO: write this in the notebook (it replaces this stub)
    raise NotImplementedError('episode_loss')


def curve_min(x, y):
    """x (n,) increasing, y (n,) a sampled U-shaped curve -> the x of its minimum: the vertex of the least-squares
    parabola through the lowest grid point and up to two neighbours on each side (5 points: a parabola through only
    3 is pulled around by the noise of a flat bottom), kept between the lowest point's two neighbours. At an end of
    the grid (the minimum may be outside the range), that end point."""
    # TODO: write this in the notebook (it replaces this stub)
    raise NotImplementedError('curve_min')


def upper_bound(x, y, tol):
    """x (n,) increasing, y (n,) a curve that is flat (within noise) and then rises; tol: how far above its floor a
    point may be and still count as flat. floor = the mean of the FLOOR_K (3) lowest y (a single lucky low point must
    not push the whole flat part out of the band). -> the first x from which EVERY point on is above floor + tol
    (a single noisy point above the band inside the flat part does not count). Values below it cannot be told apart
    from the best one; at and above it the loss is clearly worse. If the last point is within the band: inf (no bound)."""
    # TODO: write this in the notebook (it replaces this stub)
    raise NotImplementedError('upper_bound')


# --------------------------------------------------------------------------------------------- given
def choose_young(young_upper, young_min=YOUNG_MIN, young_max=YOUNG_GRID[-1]):
    """The log-middle of [young_min, young_upper]: if the data only say the true young is in that range, this value is
    at most half the range (in decades) from it, the smallest worst case of any choice. No bound (inf): young_max is used."""
    u = min(max(young_upper, young_min), young_max)
    return float(10 ** ((np.log10(young_min) + np.log10(u)) / 2))


def simulate(obs, mesh_dir, young, air=None, mass=None, damping=0.002):
    """Run the simulator with the observed hands -> (simulated points at the tracked uv (T, P, 3), all vertices (T, N, 3)).
    air / mass: None = cloth_sim.AIR / cloth_sim.MASS."""
    import cloth_sim
    sim = cloth_sim.ClothSim(mesh_dir, young=young, damping=damping, mass=mass, air=air)
    X = sim.rollout(obs['hands'], tuple(obs['grasp']), obs['t'])
    return cloth_sim.at_uv(X, sim.mesh, obs['uv']), X


def project(obs, S):
    """Simulated world points (T, P, 3) -> pixels (T, P, 2) through the recovered camera (world = R cam + tw)."""
    C = (S - obs['tw']) @ obs['R']; K = obs['K']
    return np.stack([K[0, 0] * C[..., 0] / C[..., 2] + K[0, 2], K[1, 1] * C[..., 1] / C[..., 2] + K[1, 2]], -1)


def mm_per_px(obs):
    """Pixels -> metres at the cloth's distance (median camera depth of the cloth / fx): image losses are reported in mm."""
    return float(np.nanmedian(((obs['P'] - obs['tw']) @ obs['R'])[..., 2]) / obs['K'][0, 0])


_OBS = {}
def _init(obs, sys_path):
    import sys; sys.path[:0] = [p for p in sys_path if p not in sys.path]; _OBS['o'] = obs


def _score(job):
    """One simulation -> its fit and held-out loss (mm). job = (mesh_dir, young, air, mass, space)."""
    mesh_dir, young, air, mass, space = job; obs = _OBS['o']
    S, _ = simulate(obs, mesh_dir, young, air, mass)
    O, Q = (obs['xy'], project(obs, S)) if space == 'image' else (obs['P'], S)
    k = mm_per_px(obs) if space == 'image' else 1.
    return dict(young=float(young), air=float(air), fit=episode_loss(O, Q, obs['t'], obs['fit']) * k * 1000,
                held_out=episode_loss(O, Q, obs['t'], ~obs['fit']) * k * 1000)


def sweep(obs, mesh_dir, youngs, airs, space='image', workers=None, verbose=True, label=''):
    """Simulate every (young, air) pair (lists of equal length) in parallel -> list of dicts (young, air, fit, held_out in mm)."""
    import sys, cloth_sim
    mass = cloth_sim.MASS; jobs = [(mesh_dir, float(y), float(a), mass, space) for y, a in zip(youngs, airs)]
    t0 = time.time(); out = [None] * len(jobs); workers = workers or WORKERS
    if workers <= 1:
        _init(obs, sys.path); it = map(_score, jobs)
        for i, r in enumerate(it):
            out[i] = r
            if verbose: print(f'  {label}young {r["young"]:8.3g}  air {r["air"]:5.2f}:  fit {r["fit"]:5.1f} mm  held-out {r["held_out"]:5.1f} mm   ({time.time() - t0:.0f} s)', flush=True)
        return out
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(obs, list(sys.path))) as ex:
        futs = {ex.submit(_score, j): i for i, j in enumerate(jobs)}
        from concurrent.futures import as_completed
        for f in as_completed(futs):
            r = out[futs[f]] = f.result()
            if verbose: print(f'  {label}young {r["young"]:8.3g}  air {r["air"]:5.2f}:  fit {r["fit"]:5.1f} mm  held-out {r["held_out"]:5.1f} mm   ({time.time() - t0:.0f} s)', flush=True)
    return out


def run_air(obs, mesh_dir, young=YOUNG_REF, airs=AIR_GRID, space='image', workers=None, verbose=True):
    """5a (given): young fixed, sweep the air drag -> dict(air, curve, ...). The minimum is found in log10(air)."""
    t0 = time.time(); airs = [float(a) for a in airs]
    rows = sweep(obs, mesh_dir, [young] * len(airs), airs, space, workers, verbose, 'air sweep  ')
    la = np.log10(airs); best = 10 ** curve_min(la, [r['fit'] for r in rows])
    return dict(air=float(best), log10_air=float(np.log10(best)), young_ref=float(young), curve=rows, space=space,
                at_edge=bool(best <= airs[0] or best >= airs[-1]), wall_s=round(time.time() - t0))


def noise_mm(obs, mesh_dir, air, youngs=(1e3, 1e4), space='image', workers=None, verbose=True):
    """The simulator's own noise: the same young and young x (1 + 1e-9) (a change no measurement could see) give
    losses that differ by a few mm, because where the cloth folds depends on tiny differences (chaos).
    -> (rms of those differences in mm, rows)."""
    ys = [y * f for y in youngs for f in (1., 1 + 1e-9)]
    rows = sweep(obs, mesh_dir, ys, [air] * len(ys), space, workers, verbose, 'noise      ')
    d = [rows[2 * i]['fit'] - rows[2 * i + 1]['fit'] for i in range(len(youngs))]
    return float(np.sqrt(np.mean(np.square(d)))), rows


def run_young(obs, mesh_dir, air, youngs=YOUNG_GRID, space='image', tol=None, refine=2, workers=None, verbose=True):
    """5b + 5c (given): air fixed, sweep young -> dict(young_upper, young (the chosen value), tol, curve, ...).
    tol: None = max(TOL_MIN_MM, 3 x the simulator noise measured with noise_mm).
    refine: after the grid, this many more young values evenly between the bound and the grid point below it (one
    parallel round), and the bound again: it tightens the bound from 0.5 decade steps to 0.5 / (refine + 1)."""
    t0 = time.time(); youngs = [float(y) for y in youngs]
    rows = sweep(obs, mesh_dir, youngs, [air] * len(youngs), space, workers, verbose, 'young sweep ')
    noise = None
    if tol is None:
        noise, _ = noise_mm(obs, mesh_dir, air, space=space, workers=workers, verbose=verbose); tol = max(TOL_MIN_MM, 3 * noise)
    def bound(rows):
        rows = sorted(rows, key=lambda r: r['young']); return rows, upper_bound(np.log10([r['young'] for r in rows]), [r['fit'] for r in rows], tol)
    rows, lb = bound(rows)
    if refine and np.isfinite(lb) and lb > np.log10(youngs[0]):
        below = max(np.log10(r['young']) for r in rows if np.log10(r['young']) < lb)
        extra = [10 ** (below + (lb - below) * (k + 1) / (refine + 1)) for k in range(refine)]
        rows, lb = bound(rows + sweep(obs, mesh_dir, extra, [air] * len(extra), space, workers, verbose, 'refine      '))
    ub = 10 ** lb if np.isfinite(lb) else float('inf'); fitl = [r['fit'] for r in rows]
    return dict(young_upper=float(ub), log10_young_upper=float(lb), young=choose_young(ub), log10_young=float(np.log10(choose_young(ub))),
                no_bound=bool(not np.isfinite(ub)), floor_mm=float(np.sort(np.asarray(fitl)[np.isfinite(fitl)])[:FLOOR_K].mean()), tol_mm=float(tol), noise_mm=noise, air=float(air),
                curve=rows, space=space, wall_s=round(time.time() - t0))


def run_fit(obs, mesh_dir, space='image', young_ref=YOUNG_REF, airs=AIR_GRID, youngs=YOUNG_GRID, workers=None, verbose=True):
    """The whole step 5 (given): 5a then 5b + 5c. Returns dict(air=run_air result, young=run_young result, + summary)."""
    A = run_air(obs, mesh_dir, young_ref, airs, space, workers, verbose)
    if verbose: print(f'-> air {A["air"]:.2f} N s/m^3 per m^2' + ('   (at the edge of the grid: widen AIR_GRID)' if A['at_edge'] else ''))
    Y = run_young(obs, mesh_dir, A['air'], youngs, space, workers=workers, verbose=verbose)
    if verbose: print(f'-> young <= {Y["young_upper"]:.3g} Pa (loss floor {Y["floor_mm"]:.1f} mm + tol {Y["tol_mm"]:.1f} mm)'
                      + ('   (no bound inside the grid)' if Y['no_bound'] else '') + f';  chosen young {Y["young"]:.3g} Pa')
    return dict(air=A, young=Y, air_value=A['air'], young_upper=Y['young_upper'], young_value=Y['young'], space=space)
