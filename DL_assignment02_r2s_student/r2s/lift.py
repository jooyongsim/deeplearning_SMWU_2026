"""Step 3: 2-D tracked cloth points + metric depth -> 3-D cloth points in a gravity-aligned frame, at the right scale.

Inputs (from the perception step): tracks (xy (T, P, 2) pixels, vis (T, P), uv (P, 2) where on the cloth each point is),
depth maps (T, H/2, W/2) metres from a monocular metric depth model, the camera intrinsics K, the registration
corners. Output: an `Obs` dict for fit.py.

What a monocular metric depth model gets wrong, measured on our cloth video against a RealSense:
  - its scale is off by 1.1-3.8x (Small model) / 1.3-1.7x (Base) on the cloth, and it CHANGES from frame to frame
  - with one correct scale per frame, the depth inside the cloth is right to ~3.5 % (Base, ~2 cm at 0.65 m)
So the scale is fixed in every frame with what we know: the cloth's size. Neighbouring tracked points are a known
distance apart on the cloth (their uv difference x the cloth's width / height); their 3-D distance should match.

Your TODO functions (python r2s/check.py / the notebook's test cell checks them):
  backproject       pixel + depth -> camera-frame 3-D point (pinhole K)
  per_frame_scale   one scale per frame from the known distances between neighbouring tracked points
  floor_frame       the floor is a plane: its normal is "up"; rotate so z is up and the floor is at z = 0

Gravity from vertical lines (optional, given): the floor plane needs a wide band of floor in the bottom of the video;
when the cloth comes down close to the bottom edge, only a thin strip is left and the plane tilts (T01: 17 deg, camera
"height" -2 cm). Lines that are vertical in the room (door frames, furniture edges) all point along gravity, which
needs only K: `vertical_candidates` finds them, `up_from_lines` turns the chosen ones into the 'up' direction, and
`lift_take(..., up=up)` uses it instead of the floor's normal.
"""
import numpy as np
import cv2


def sample(depth, xy, r=2):
    """Depth at video pixels xy (P, 2) (depth maps are at half the video resolution): a low percentile (25 %) of the
    (2r+1)^2 window. Points on the cloth's edge would otherwise pick up the background behind it (metres farther);
    the cloth is in front of the background, and a low percentile (not the minimum) is not pulled by pixel noise."""
    D = np.asarray(depth, np.float32); H, W = D.shape; p = np.round(np.asarray(xy, float) / 2.).astype(int)
    x = np.clip(p[:, 0], r, W - r - 1); y = np.clip(p[:, 1], r, H - r - 1)
    win = np.stack([D[y + dy, x + dx] for dy in range(-r, r + 1) for dx in range(-r, r + 1)], 1)
    win = np.where(win > 0, win, np.nan)
    return np.nanpercentile(win, 25, axis=1)


def backproject(xy, z, K):
    """xy (..., 2) pixels, z (...) depth along the optical axis (m), K 3x3 -> (..., 3) camera-frame points
    (x right, y down, z forward)."""
    # TODO: write this in the notebook (it replaces this stub)
    raise NotImplementedError('backproject')


def neighbour_pairs(uv, width, height):
    """Pairs (i, j) of grid points that are neighbours along u or v, and their distance on the flat cloth (m)."""
    du, dv = np.diff(np.unique(uv[:, 0])).min(), np.diff(np.unique(uv[:, 1])).min(); pairs, L = [], []
    for i in range(len(uv)):
        for j in range(i + 1, len(uv)):
            d = np.abs(uv[i] - uv[j])
            if (np.isclose(d[0], du) and np.isclose(d[1], 0)) or (np.isclose(d[1], dv) and np.isclose(d[0], 0)):
                pairs.append((i, j)); L.append(np.hypot(d[0] * width, d[1] * height))
    return np.array(pairs), np.array(L)


def per_frame_scale(X, vis, pairs, L, min_pairs=8):
    """X (T, P, 3) camera-frame points from the raw metric depth, vis (T, P), neighbour pairs (Q, 2) and their known
    lengths L (Q,). Returns s (T,): for each frame the median over the pairs with both points visible of
    known length / measured 3-D length; NaN where fewer than min_pairs pairs are visible."""
    # TODO: write this in the notebook (it replaces this stub)
    raise NotImplementedError('per_frame_scale')


def fit_plane_ransac(P, iters=500, tol=.02, rng=None):
    """Plane through most of the points: (unit normal n, a point c on it, inlier fraction)."""
    rng = rng or np.random.default_rng(0); best = (0, None)
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]; n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9: continue
        n /= np.linalg.norm(n); inl = np.abs((P - a) @ n) < tol
        if inl.sum() > best[0]: best = (inl.sum(), inl)
    Q = P[best[1]]; c = Q.mean(0); n = np.linalg.svd(Q - c)[2][-1]
    return n, c, best[1].mean()


def floor_frame(P_floor):
    """Camera-frame floor points -> (R, t, inlier fraction) with world = R @ cam + t: z up, floor at z = 0, x along
    the camera's x projected on the floor. Use fit_plane_ransac; camera y points down, so "up" has negative camera y."""
    # TODO: write this in the notebook (it replaces this stub)
    raise NotImplementedError('floor_frame')


def frame_from_up(up):
    """'up' in camera coordinates -> R (rows x, y, z = up) with x along the camera's x projected off 'up' (as floor_frame)."""
    n = np.asarray(up, float) / np.linalg.norm(up); n = -n if n[1] > 0 else n
    x = np.array([1., 0, 0]) - n[0] * n; x /= np.linalg.norm(x); return np.stack([x, np.cross(n, x), n])


def vertical_candidates(img, min_len=60, max_deg=20, top=20):
    """Near-vertical straight segments in an RGB frame (N, 4: x1 y1 x2 y2), longest first: candidates for room-vertical lines."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    try: segs = cv2.createLineSegmentDetector().detect(g)[0].reshape(-1, 4).astype(float)
    except Exception: segs = cv2.HoughLinesP(cv2.Canny(g, 50, 150), 1, np.pi / 180, 60, minLineLength=min_len, maxLineGap=5).reshape(-1, 4).astype(float)
    d = segs[:, 2:] - segs[:, :2]; L = np.hypot(d[:, 0], d[:, 1]); ang = np.degrees(np.arctan2(np.abs(d[:, 0]), np.abs(d[:, 1])))
    keep = (L >= min_len) & (ang <= max_deg); return segs[keep][np.argsort(-L[keep])][:top]


def up_from_lines(segs, K):
    """Image segments (N >= 2, 4) of lines that are vertical in the room -> ('up' unit vector in camera coordinates,
    residual of each line in degrees). Each line and the camera centre span a plane with normal K^T l (l = the line
    through the two end points); gravity lies in all of them: the smallest singular vector of the stacked normals."""
    M = np.array([K.T @ np.cross([x1, y1, 1.], [x2, y2, 1.]) for x1, y1, x2, y2 in np.asarray(segs, float)])
    M /= np.linalg.norm(M, axis=1, keepdims=True); d = np.linalg.svd(M)[2][-1]; d = -d if d[1] > 0 else d    # camera y is down
    return d, np.degrees(np.arcsin(np.clip(np.abs(M @ d), 0, 1)))


def smooth(s, k=5):
    """Median filter over k frames, NaNs filled from the neighbours (depth scale changes slowly; single frames are noisy)."""
    s = np.asarray(s, float).copy(); idx = np.arange(len(s)); ok = np.isfinite(s)
    s = np.interp(idx, idx[ok], s[ok]); h = k // 2
    return np.array([np.median(s[max(0, i - h):i + h + 1]) for i in range(len(s))])


def smooth_traj(Y, k=7):
    """Centred moving average over k frames along axis 0 (NaNs filled from the neighbours first)."""
    Y = np.asarray(Y, float).copy(); T = len(Y); idx = np.arange(T); flat = Y.reshape(T, -1)
    for j in range(flat.shape[1]):
        ok = np.isfinite(flat[:, j])
        if ok.any() and not ok.all(): flat[:, j] = np.interp(idx, idx[ok], flat[ok, j])
    h = k // 2; out = np.array([flat[max(0, i - h):i + h + 1].mean(0) for i in range(T)])
    return out.reshape(Y.shape)


def lift_take(tracks, depth, K, reg, mesh, t, floor_rows=.6, depth_frames=None, up=None, hand_smooth=7):
    """The whole step (given). tracks: dict(frames, xy, vis, uv, corner_ids); depth (Td, H/2, W/2) for depth_frames
    (default: the same frames as the tracks); reg: register dict (frame, corners_px, held, fit_end); mesh: load_mesh();
    t: time of every video frame (camera.json). up: 'up' in camera coordinates (up_from_lines) to use instead of the
    floor plane's normal (the floor then only sets the height z = 0); hand_smooth: frames in the moving average of the
    hand trajectories (depth flicker makes the hands jitter, mostly along the camera axis, and shakes the simulated
    cloth). Returns obs dict and a dict of diagnostics."""
    frames, xy, vis, uv = tracks['frames'], tracks['xy'], tracks['vis'].astype(bool), tracks['uv']
    D = np.asarray(depth, np.float32); dfr = np.asarray(depth_frames if depth_frames is not None else frames)
    di = np.array([int(np.argmin(np.abs(dfr - f))) for f in frames])
    z = np.array([sample(D[di[k]], xy[k]) for k in range(len(frames))])
    Xr = backproject(xy, z, K)                                          # raw metric depth, camera frame
    pairs, L = neighbour_pairs(uv, mesh['width'], mesh['height'])
    s_raw = per_frame_scale(Xr, vis, pairs, L); s = smooth(s_raw)
    Xc = Xr * s[:, None, None]
    # floor: the camera does not move, so the floor's depth is the same in every frame: the median over all frames
    # (each at its own scale) removes most of the per-frame noise. Floor pixels: the bottom rows the cloth never enters.
    H, W = D.shape[1:]; Dm = np.median(np.stack([D[di[k]] * s[k] for k in range(0, len(frames), 3)]), 0)
    ybot = np.nanmax(np.where(vis, xy[..., 1], np.nan)) / 2 + 8; y0 = int(min(max(ybot, floor_rows * H), H - 20))
    yy, xx = np.mgrid[0:H, 0:W]; sel = (yy > y0) & (Dm > 0)
    Pf = backproject(np.stack([xx[sel] * 2., yy[sel] * 2.], -1), Dm[sel], K)
    Pf = Pf[np.random.default_rng(0).choice(len(Pf), min(len(Pf), 20000), replace=False)]
    if up is None: R, tw, frac = floor_frame(Pf); gravity = 'floor plane'
    else:                                                                # gravity from vertical lines; the floor gives only the height
        R = frame_from_up(up); h = Pf @ R[2]; h0 = np.median(h); tw = np.array([0., 0., -h0]); frac = float(np.mean(np.abs(h - h0) < .02)); gravity = 'vertical lines'
    X = Xc @ R.T + tw
    P = smooth_traj(np.where(vis[..., None], X, np.nan), k=5); P[~vis] = np.nan        # a little temporal smoothing, hidden stay hidden
    held = reg['held']; cid = tracks['corner_ids']; hands = X[:, [cid[held[0]], cid[held[1]]]]          # tracked even when hidden by fingers
    hands_raw = hands; hands = smooth_traj(hands, k=hand_smooth)   # depth noise makes the hands jitter; jittery hands shake the simulated cloth
    xy_obs = np.where(vis[..., None], xy, np.nan).astype(np.float32)                      # the tracked pixels (seen only)
    obs = dict(t=np.asarray(t)[frames], frames=frames, P=P.astype(np.float32), uv=uv, hands=hands.astype(np.float32), xy=xy_obs, K=np.asarray(K, float),
               grasp=np.asarray(mesh['corners'])[held], fit=frames <= reg.get('fit_end', frames[-1]), R=R, tw=tw, scale=s)
    diag = dict(scale_raw=s_raw, scale=s, floor_inliers=frac, camera_height=float(tw[2]), gravity=gravity, floor_rows=int(H - y0),
                tilt_deg=float(np.degrees(np.arccos(np.clip(-R[2][1], -1, 1)))), cloth_height=mesh['height'],
                hand_jitter_mm=float(np.sqrt(np.nanmean((hands_raw - hands) ** 2)) * 1000),
                hand_separation=np.linalg.norm(hands[:, 0] - hands[:, 1], axis=1), seen=float(vis.mean()))
    return obs, diag
