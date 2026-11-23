"""Pictures and videos for every step (given in full). Images are RGB uint8."""
import cv2
import numpy as np


def sheet(imgs, cols=4, w=480, labels=None):
    """A grid of images, each resized to width w, optional text labels."""
    out = []
    for i, im in enumerate(imgs):
        im = cv2.resize(np.ascontiguousarray(im), (w, int(im.shape[0] * w / im.shape[1])))
        if labels is not None:
            for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)): cv2.putText(im, str(labels[i]), (8, 26), cv2.FONT_HERSHEY_SIMPLEX, .7, col, th, cv2.LINE_AA)
        out.append(im)
    out += [np.zeros_like(out[0])] * (-len(out) % cols)
    return np.vstack([np.hstack(out[i:i + cols]) for i in range(0, len(out), cols)])


def show(img, title='', size=(10, 5.6)):
    import matplotlib.pyplot as plt
    plt.figure(figsize=size); plt.imshow(img); plt.title(title); plt.axis('off'); plt.show()


def overlay_mask(img, mask, color=(0, 110, 0)):
    v = img.copy(); v[mask] = (.55 * v[mask] + color).clip(0, 255).astype(np.uint8); return v


def draw_points(img, xy, vis=None, held=(), r=4):
    """Tracked points: green seen, red hidden, blue the held corners."""
    im = img.copy(); vis = np.ones(len(xy), bool) if vis is None else vis
    for j, (p, ok) in enumerate(zip(xy, vis)):
        if not np.isfinite(p).all(): continue
        cv2.circle(im, (int(p[0]), int(p[1])), r + 3 if j in held else r, (0, 90, 255) if j in held else ((40, 200, 40) if ok else (230, 40, 40)), -1)
    return im


def draw_corners(img, C, color=(255, 0, 255), names=True):
    im = img.copy(); cv2.polylines(im, [np.asarray(C, np.int32)], True, color, 2)
    for i, c in enumerate(np.asarray(C)):
        cv2.circle(im, (int(c[0]), int(c[1])), 7, color, -1)
        if names: cv2.putText(im, f'c{i}', (int(c[0]) + 9, int(c[1]) + 6), cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2, cv2.LINE_AA)
    return im


def depth_color(d, lo=.3, hi=3.):
    import matplotlib.cm as cm
    return (cm.turbo(np.clip((np.asarray(d, float) - lo) / (hi - lo), 0, 1))[..., :3] * 255).astype(np.uint8)


def write_video(path, frames, fps=30):
    import imageio_ffmpeg
    H, W = frames[0].shape[:2]; w = imageio_ffmpeg.write_frames(str(path), (W, H), fps=fps, codec='libx264', pix_fmt_out='yuv420p', quality=7, macro_block_size=2)
    w.send(None)
    for f in frames: w.send(np.ascontiguousarray(f).tobytes())
    w.close(); return path


def project(obs, X):
    """World points (..., 3) -> pixels (..., 2) through the recovered camera."""
    C = (X - obs['tw']) @ obs['R']; K = obs['K']
    return np.stack([K[0, 0] * C[..., 0] / C[..., 2] + K[0, 2], K[1, 1] * C[..., 1] / C[..., 2] + K[1, 2]], -1)


def overlay_sim(frames, obs, X, F=None, every=1, label=''):
    """The simulated cloth (vertices X (T, N, 3), triangles F) drawn as a wireframe on the video frames of the take,
    with the tracked points (green) and the hands (blue)."""
    out = []
    for k in range(0, len(obs['frames']), every):
        im = frames[obs['frames'][k]].copy(); p = project(obs, X[k])
        if F is not None:
            for f in F[::2]: cv2.polylines(im, [p[f].astype(np.int32)], True, (255, 150, 0), 1, cv2.LINE_AA)
        im = draw_points(im, obs['xy'][k], np.isfinite(obs['xy'][k]).all(1), r=3)
        for h in project(obs, obs['hands'][k]): cv2.circle(im, (int(h[0]), int(h[1])), 9, (0, 90, 255), 2)
        txt = f'{label}  frame {obs["frames"][k]}  {"fit" if obs["fit"][k] else "held-out"}'
        for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)): cv2.putText(im, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, col, th, cv2.LINE_AA)
        out.append(im)
    return out


def plot_lift(obs, diag, title=''):
    """Per-frame scale, hand separation, and the 3-D points of a few frames."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 4.2))
    ax = fig.add_subplot(1, 3, 1); ax.plot(obs['frames'], diag['scale_raw'], '.', ms=3, alpha=.5, label='per frame'); ax.plot(obs['frames'], diag['scale'], label='smoothed')
    ax.set(xlabel='frame', ylabel='scale (known / measured)', title='depth scale from the cloth'); ax.legend()
    ax = fig.add_subplot(1, 3, 2); ax.plot(obs['frames'], diag['hand_separation'] * 100); ax.axhline(diag.get('cloth_height', .32) * 100, ls='--', c='k', lw=.8)
    ax.set(xlabel='frame', ylabel='cm', title='distance between the hands (32 cm = edge length)')
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    for k in np.linspace(0, len(obs['frames']) - 1, 4).astype(int):
        P = obs['P'][k]; ok = np.isfinite(P).all(1); ax.scatter(P[ok, 0], P[ok, 1], P[ok, 2], s=4, label=f'frame {obs["frames"][k]}')
        ax.scatter(*obs['hands'][k].T, s=40, marker='x', c='k')
    ax.set(xlabel='x', ylabel='y', zlabel='z (up, m)', title='cloth points (world, floor z = 0)'); ax.legend(fontsize=7)
    fig.suptitle(title); plt.tight_layout(); plt.show()


def plot_air(A, title='', ax=None, show=True):
    """5a: loss vs air drag (young fixed) with the chosen minimum. A = fit.run_air(...) (or fit.run_fit(...)['air'])."""
    import matplotlib.pyplot as plt
    c = A['curve']; a = [r['air'] for r in c]
    if ax is None: plt.figure(figsize=(6, 3.8)); ax = plt.gca()
    ax.semilogx(a, [r['fit'] for r in c], 'o-', label='fit frames'); ax.semilogx(a, [r['held_out'] for r in c], 's--', alpha=.6, label='held-out frames')
    ax.axvline(A['air'], c='k', lw=.8); ax.set_xlabel('air drag (N s/m$^3$ per m$^2$ of cloth)'); ax.set_ylabel('mean distance (mm)')
    ax.set_title(title or f'5a  air = {A["air"]:.2f}   (young fixed at {A["young_ref"]:.2g})'); ax.legend()
    if show: plt.tight_layout(); plt.show()


def plot_young(Y, title='', ax=None, show=True):
    """5b-c: loss vs young (air fixed): the flat floor, the tolerance band, the upper bound and the chosen value."""
    import matplotlib.pyplot as plt
    c = Y['curve']; y = [r['young'] for r in c]
    if ax is None: plt.figure(figsize=(6, 3.8)); ax = plt.gca()
    ax.axhspan(Y['floor_mm'], Y['floor_mm'] + Y['tol_mm'], color='g', alpha=.12, label=f'floor + tol ({Y["tol_mm"]:.1f} mm)')
    ax.semilogx(y, [r['fit'] for r in c], 'o-', label='fit frames'); ax.semilogx(y, [r['held_out'] for r in c], 's--', alpha=.6, label='held-out frames')
    ub = Y['young_upper'] if np.isfinite(Y['young_upper']) else max(y)
    ax.axvline(ub, c='r', lw=1.2, label=f'upper bound {Y["young_upper"]:.2g}' if np.isfinite(Y['young_upper']) else 'no upper bound in the grid')
    ax.axvline(Y['young'], c='k', lw=.8, ls=':', label=f'chosen {Y["young"]:.2g}'); ax.axvspan(min(y), ub, color='0.5', alpha=.06)
    ax.set_xlabel('young (bending stiffness, Pa)'); ax.set_ylabel('mean distance (mm)')
    ax.set_title(title or f'5b  young <= {Y["young_upper"]:.2g}   (air fixed at {Y["air"]:.2f})'); ax.legend(fontsize=8)
    if show: plt.tight_layout(); plt.show()


def plot_fit(result, title=''):
    """Both sub-steps side by side. result = fit.run_fit(...)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    plot_air(result['air'], ax=ax[0], show=False); plot_young(result['young'], ax=ax[1], show=False)
    if title: fig.suptitle(title)
    plt.tight_layout(); plt.show()
