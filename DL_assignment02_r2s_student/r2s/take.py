"""Step 1: a phone video -> frames, time stamps, camera intrinsics, the registration corners (given in full).

    frames, t = load_video('take1.mp4', width=960)          # landscape, resized, real time stamps
    K = calibrate_checkerboard('board.mp4', width=960)     # printed 9x6 checkerboard, 25 mm squares
    C = auto_corners(mask)                                  # 4 cloth corners c0..c3 from the registration mask
    f = focal_from_corners(C, (W, H))                       # no checkerboard: the cloth's vanishing points
"""
import cv2
import numpy as np


def load_video(path, width=960, start=0, end=None):
    """RGB frames (T, H, W, 3) uint8 resized to `width`, and their time stamps (s, from the container: phones record
    variable frame rates)."""
    cap = cv2.VideoCapture(str(path)); frames, t = [], []; k = 0
    while True:
        ok, im = cap.read()
        if not ok or (end is not None and k > end): break
        if k >= start:
            if im.shape[0] > im.shape[1]: raise ValueError('portrait video: film in landscape (docs/FILMING.md)')
            s = width / im.shape[1]; h = int(round(im.shape[0] * s / 2) * 2)
            frames.append(cv2.resize(im, (width, h), interpolation=cv2.INTER_AREA)[..., ::-1]); t.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        k += 1
    t = np.array(t); return np.stack(frames), t - t[0]


def calibrate_checkerboard(path, width=960, inner=(9, 6), square_m=.025, every=10, max_views=40):
    """K (3x3) for frames resized to `width`, from a video of the printed checkerboard; also the RMS reprojection
    error (px, at the original resolution) and the number of board views used."""
    nx, ny = inner; obj = np.zeros((nx * ny, 3), np.float32); obj[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * square_m
    cap = cv2.VideoCapture(str(path)); O, I = [], []; k = 0; size = None
    while len(I) < max_views:
        ok, im = cap.read()
        if not ok: break
        if k % every == 0:
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); size = g.shape[::-1]
            ok2, c = cv2.findChessboardCorners(g, (nx, ny), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ok2: O.append(obj); I.append(cv2.cornerSubPix(g, c, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)))
        k += 1
    if len(I) < 8: raise RuntimeError(f'only {len(I)} views of the whole board: film it closer and more slowly')
    err, K, dist, _, _ = cv2.calibrateCamera(O, I, size, None, None)
    s = width / size[0]; K = K.copy(); K[:2] *= s
    return K, float(err), len(I)


def focal_from_corners(C, size):
    """A rectangle seen in perspective: the vanishing points of its two pairs of opposite sides are orthogonal 3-D
    directions, so f^2 = -(v1 - c).(v2 - c), c = the image centre. None when the sides are (nearly) parallel."""
    L = [np.cross(np.r_[C[i], 1.], np.r_[C[(i + 1) % 4], 1.]) for i in range(4)]
    v1, v2 = np.cross(L[0], L[2]), np.cross(L[1], L[3]); c = np.array(size, float) / 2
    if abs(v1[2]) < 1e-9 or abs(v2[2]) < 1e-9: return None
    f2 = -(v1[:2] / v1[2] - c) @ (v2[:2] / v2[2] - c)
    return float(np.sqrt(f2)) if f2 > 0 else None


def fit_quad(mask):
    """4 corners of a mask's outline: a quadrilateral approximation of its convex hull."""
    cnt = max(cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)[:, 0]
    hull = cv2.convexHull(cnt); peri = cv2.arcLength(hull, True)
    for eps in np.linspace(.005, .1, 200):
        q = cv2.approxPolyDP(hull, eps * peri, True)[:, 0]
        if len(q) == 4: return q.astype(float)
    raise RuntimeError('the mask is not a quadrilateral')


def auto_corners(mask):
    """The cloth's corners in the mesh order c0..c3 from the registration-frame mask: the two upper ones are held
    (c0 = left hand, c3 = right hand), c1 below c0, c2 below c3 (docs/FILMING.md)."""
    Q = fit_quad(mask); top = Q[np.argsort(Q[:, 1])[:2]]; bot = Q[np.argsort(Q[:, 1])[2:]]
    c0, c3 = top[np.argsort(top[:, 0])]; c1, c2 = bot[np.argsort(bot[:, 0])]
    return np.array([c0, c1, c2, c3])
