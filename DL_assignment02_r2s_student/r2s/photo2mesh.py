#!/usr/bin/env python3
"""Milestone 1: one photo of a cloth lying flat on a table -> a flat, textured triangle mesh the simulator loads.

    python mesh/photo2mesh.py --photo my_cloth.jpg --side_cm 32 --out assets/my_cloth
    python mesh/photo2mesh.py --photo my_cloth.jpg --side_cm 32 --out assets/my_cloth --seg sam       # GroundingDINO + SAM 2.1
    python mesh/photo2mesh.py --photo my_cloth.jpg --side_cm 32 --out assets/my_cloth --seg click     # click the 4 corners
    python mesh/photo2mesh.py ... --seg corners --corners "560,620;2790,740;2140,2700;80,1980"         # corner pixels given
    python mesh/photo2mesh.py ... --seg mask        # reuse --out/mask.png (e.g. made on Colab, or edited by hand)

Why not a generic image-to-3D model (TRELLIS, ...): for a thin sheet they return a thick or closed blob, and a cloth
simulator wants an open, single-layer grid. The cloth is a sewn rectangle lying flat, so one photo gives everything:

  1. segment    the cloth mask (SAM, or OpenCV GrabCut, or the 4 corners by hand)
  2. fit_quad   fit a line to each of the 4 sides of the mask outline; corners = line intersections
  3. camera     the two vanishing points of opposite sides are orthogonal directions in 3-D, which fixes the focal
                length: f^2 = -(v1 - c).(v2 - c)  (principal point c at the image centre, square pixels; no EXIF needed)
  4. rectify    homography + K -> the true aspect ratio; one measured side (--side_cm, the LONGER side) -> metres;
                the photo warped onto the cloth plane -> texture.png (straight-on view of the print)
  5. grid       a regular grid at --spacing (default 2 cm, what the simulator runs at), alternating diagonals, UVs
  6. check      the mesh drawn back onto the photo with the fitted camera, IoU against the mask

Writes into --out:
  cloth_mesh.npz   vertices (N,3) m, z up, centred, on z = 0; faces (M,3); uv (N,2); corners (4,) vertex ids
                   c0 (-x,-y), c1 (+x,-y), c2 (+x,+y), c3 (-x,+y) (counter-clockwise); grid_shape (ny, nx); width_m, height_m
  texture.png      rectified photo of the cloth (2 px/mm)
  cloth.obj, cloth.mtl   the same mesh for any 3-D viewer (Blender, MeshLab)
  mask_overlay.jpg, corners.jpg, reprojection.jpg, mesh_report.json   the checks (Milestone 1 deliverables)
"""
import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np


# ---------- image I/O that works with non-ASCII paths (cv2.imread / imwrite fail silently on Windows) ----------
def imread(path):
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(path)
    return img


def imwrite(path, img):
    ok, buf = cv2.imencode(Path(path).suffix, img)
    if not ok: raise IOError(f'could not encode {path}')
    Path(path).write_bytes(buf.tobytes())


# ---------- 1. segmentation ----------
def segment_sam(bgr, prompt='cloth.'):
    """GroundingDINO (text -> box) then SAM 2.1 (box -> mask). Needs torch + transformers (requirements-mesh.txt);
    downloads ~1 GB of weights the first time. A GPU (Colab) makes it fast; the CPU takes about a minute."""
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); W, H = img.size
    gd_id = 'IDEA-Research/grounding-dino-tiny'
    proc = AutoProcessor.from_pretrained(gd_id); gd = AutoModelForZeroShotObjectDetection.from_pretrained(gd_id).to(dev)
    inp = proc(images=img, text=prompt, return_tensors='pt').to(dev)
    with torch.no_grad(): out = gd(**inp)
    det = proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.3, text_threshold=0.25, target_sizes=[(H, W)])[0]
    if len(det['scores']) == 0: raise RuntimeError(f'GroundingDINO found nothing for {prompt!r}; try --prompt "white cloth." or --seg click')
    box = det['boxes'][int(det['scores'].argmax())].tolist()
    print(f'  GroundingDINO box {[round(b) for b in box]} score {float(det["scores"].max()):.2f}')
    sam_id = 'facebook/sam2.1-hiera-small'
    sp = Sam2Processor.from_pretrained(sam_id); sam = Sam2Model.from_pretrained(sam_id).to(dev)
    inp = sp(images=img, input_boxes=[[box]], return_tensors='pt').to(dev)
    with torch.no_grad(): out = sam(**inp, multimask_output=False)
    m = sp.post_process_masks(out.pred_masks.cpu(), inp['original_sizes'].cpu())[0]
    return clean_mask(m[0, 0].numpy().astype(np.uint8))


def segment_grabcut(bgr, box=None):
    """OpenCV GrabCut, no deep learning: works when the cloth stands out from the table and fills the middle of the
    photo. box (x0, y0, x1, y1) in pixels: where the cloth is (default: the photo minus a 10 % border)."""
    H, W = bgr.shape[:2]; s = 800 / max(H, W); small = cv2.resize(bgr, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    rect = (int(w * .1), int(h * .1), int(w * .8), int(h * .8)) if box is None else \
        (int(box[0] * s), int(box[1] * s), int((box[2] - box[0]) * s), int((box[3] - box[1]) * s))
    # colour prior: the cloth's colour is sampled in the middle of the box; similar pixels start as "probably cloth",
    # the rest as "probably background", outside the box as background (a plain rectangle start leaks into the table)
    x, y, rw, rh = rect; lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = np.median(lab[y + rh * 2 // 5:y + rh * 3 // 5, x + rw * 2 // 5:x + rw * 3 // 5].reshape(-1, 3), 0)
    dist = np.linalg.norm(lab - ref, axis=2); thr = max(np.percentile(dist[y:y + rh, x:x + rw], 40) * 1.5, 12)
    mask = np.full((h, w), cv2.GC_BGD, np.uint8); mask[y:y + rh, x:x + rw] = cv2.GC_PR_BGD
    inside = np.zeros((h, w), bool); inside[y:y + rh, x:x + rw] = True
    mask[inside & (dist < thr)] = cv2.GC_PR_FGD; mask[inside & (dist < thr / 3)] = cv2.GC_FGD
    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(small, mask, None, bg, fg, 8, cv2.GC_INIT_WITH_MASK)
    m = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    return clean_mask(cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST))


def clean_mask(mask):
    """Largest connected part, holes filled (the printed pattern can punch holes in the mask)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if n < 2: raise RuntimeError('empty mask')
    mask = (lab == 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cv2.drawContours(np.zeros_like(mask), cnts, -1, 1, cv2.FILLED)


def click_corners(bgr):
    """Click the 4 corners (any order) in a window; Enter to accept, Backspace to undo."""
    H, W = bgr.shape[:2]; s = min(1., 900 / max(H, W)); view = cv2.resize(bgr, (int(W * s), int(H * s))); pts = []
    def cb(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4: pts.append((x / s, y / s))
    cv2.namedWindow('click the 4 cloth corners, Enter = done'); cv2.setMouseCallback('click the 4 cloth corners, Enter = done', cb)
    while True:
        v = view.copy()
        for p in pts: cv2.circle(v, (int(p[0] * s), int(p[1] * s)), 6, (0, 0, 255), -1)
        if len(pts) > 1: cv2.polylines(v, [(np.array(pts) * s).astype(np.int32)], len(pts) == 4, (0, 255, 0), 2)
        cv2.imshow('click the 4 cloth corners, Enter = done', v); k = cv2.waitKey(30)
        if k in (13, 10) and len(pts) == 4: break
        if k == 8 and pts: pts.pop()
    cv2.destroyAllWindows(); return order_corners(np.array(pts, float))


# ---------- 2. quadrilateral fit ----------
def order_corners(q):
    """top-left, top-right, bottom-right, bottom-left (in the photo)."""
    q = np.asarray(q, float); c = q.mean(0); a = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]); q = q[np.argsort(a)]   # clockwise on screen from -pi
    start = np.argmin(q.sum(1)); return np.roll(q, -start, axis=0)


def fit_quad(mask):
    """Corners of the cloth: a quad from the convex hull, then a robust line fit to each side's middle 70 % (the
    curled hems and rounded corners would pull a plain fit), corners = intersections of neighbouring lines."""
    cnt = max(cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)[:, 0]
    hull = cv2.convexHull(cnt); peri = cv2.arcLength(hull, True); quad = None
    for eps in np.linspace(.005, .1, 200):
        q = cv2.approxPolyDP(hull, eps * peri, True)[:, 0]
        if len(q) == 4: quad = order_corners(q); break
    if quad is None: raise RuntimeError('the mask outline is not a quadrilateral; use --seg click')
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]; ab = b - a; rel = cnt - a
        t = rel @ ab / (ab @ ab); dist = np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / np.linalg.norm(ab)
        pts = cnt[(t > .15) & (t < .85) & (dist < .03 * np.linalg.norm(ab))].astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, .01, .01).ravel()
        lines.append(np.cross([x0, y0, 1.], [x0 + vx, y0 + vy, 1.]))
    corners = np.array([np.cross(lines[i - 1], lines[i]) for i in range(4)]); corners = corners[:, :2] / corners[:, 2:]
    # a side whose fit went wrong (e.g. a curled hem tip) puts its corners far away: keep the hull's corner there
    diag = np.linalg.norm(quad[2] - quad[0])
    bad = ~np.isfinite(corners).all(1) | (np.linalg.norm(corners - quad, axis=1) > .08 * diag)
    corners[bad] = quad[bad]
    corners = polish(corners, mask)
    return corners, lines_of(corners)


def polish(corners, mask, size=750):
    """Move each corner a few pixels at a time while the quad's IoU with the mask improves (coarse to fine)."""
    s = size / max(mask.shape); m = cv2.resize(mask.astype(np.uint8), None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST).astype(bool)
    def iou(c):
        q = np.zeros(m.shape, np.uint8); cv2.fillPoly(q, [np.round(c).astype(np.int32)], 1); q = q.astype(bool)
        return (q & m).sum() / max((q | m).sum(), 1)
    c = corners * s; best = iou(c)
    for step in (8, 4, 2, 1, .5):
        improved = True
        while improved:
            improved = False
            for i in range(4):
                for d in ((step, 0), (-step, 0), (0, step), (0, -step)):
                    t = c.copy(); t[i] += d; v = iou(t)
                    if v > best + 1e-6: c, best, improved = t, v, True
    print(f'  quad vs mask IoU {best:.3f}')
    return c / s


def lines_of(corners):
    return [np.cross(np.r_[corners[i], 1.], np.r_[corners[(i + 1) % 4], 1.]) for i in range(4)]


# ---------- 3-4. camera and metric rectangle ----------
def fit_plane(corners, lines, image_size, side_cm, f_px=None):
    W, H = image_size; c = np.array([W / 2, H / 2])
    def vp(l1, l2): v = np.cross(l1, l2); return v[:2] / v[2] if abs(v[2]) > 1e-12 else v[:2] * 1e12
    if f_px is None:
        f2 = -(vp(lines[0], lines[2]) - c) @ (vp(lines[1], lines[3]) - c)
        if f2 <= 0 or not np.isfinite(f2):       # sides (almost) parallel in the photo: the view is too straight-on
            f_px = 1.2 * max(W, H); print(f'  vanishing points uninformative; assuming f = {f_px:.0f} px')
        else: f_px = float(np.sqrt(f2))
    K = np.array([[f_px, 0, c[0]], [0, f_px, c[1]], [0, 0, 1.]])
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), corners.astype(np.float32))
    B = np.linalg.inv(K) @ Hm; w_rel, h_rel = np.linalg.norm(B[:, 0]), np.linalg.norm(B[:, 1]); aspect = w_rel / h_rel
    L = side_cm / 100; w_m, h_m = (L, L / aspect) if aspect >= 1 else (L * aspect, L)
    obj = np.array([[0, 0, 0], [w_m, 0, 0], [w_m, h_m, 0], [0, h_m, 0]], float)
    ok, rvec, tvec = cv2.solvePnP(obj, corners, K, None, flags=cv2.SOLVEPNP_IPPE)
    R = cv2.Rodrigues(rvec)[0]; proj = cv2.projectPoints(obj, rvec, tvec, K, None)[0][:, 0]
    ortho = float(np.degrees(np.arccos(np.clip((B[:, 0] / w_rel) @ (B[:, 1] / h_rel), -1, 1))))
    return dict(K=K, R=R, t=tvec.ravel(), width_m=w_m, height_m=h_m, aspect=aspect, f_px=f_px,
                f_35mm=f_px / np.hypot(W, H) * 43.27, reproj_px=float(np.abs(proj - corners).max()), side_angle_deg=ortho)


def rectify(bgr, mask, P, px_per_mm=2.):
    """Straight-on texture of the cloth plane: plane metres -> image pixels is K [r1 r2 t]."""
    tw, th = int(round(P['width_m'] * 1000 * px_per_mm)), int(round(P['height_m'] * 1000 * px_per_mm))
    A = np.diag([1 / (1000 * px_per_mm), 1 / (1000 * px_per_mm), 1.])
    M = P['K'] @ np.column_stack([P['R'][:, 0], P['R'][:, 1], P['t']]) @ A
    tex = cv2.warpPerspective(bgr, M, (tw, th), flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP)
    tm = cv2.warpPerspective(mask.astype(np.uint8) * 255, M, (tw, th), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 127
    tm = cv2.erode(tm.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    fill = float(tm.mean())
    tex = cv2.inpaint(tex, (~tm).astype(np.uint8) * 255, 15, cv2.INPAINT_TELEA)   # curled hems / rounded corners: no table on the cloth
    return tex, fill


# ---------- 5. grid mesh ----------
def grid_mesh(w, h, spacing):
    """Regular grid centred at the origin, z = 0, alternating diagonals (no preferred shear direction).
    Row j runs along +x at y = -h/2 + j dy. Faces are counter-clockwise seen from +z."""
    nx, ny = max(2, int(round(w / spacing)) + 1), max(2, int(round(h / spacing)) + 1)
    X, Y = np.meshgrid(np.linspace(-w / 2, w / 2, nx), np.linspace(-h / 2, h / 2, ny))
    V = np.c_[X.ravel(), Y.ravel(), np.zeros(nx * ny)]; idx = np.arange(nx * ny).reshape(ny, nx); F = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a, b, c, d = idx[j, i], idx[j, i + 1], idx[j + 1, i + 1], idx[j + 1, i]
            F += [[a, b, c], [a, c, d]] if (i + j) % 2 == 0 else [[a, b, d], [b, c, d]]
    F = np.array(F, np.int32)
    uv = np.c_[(V[:, 0] + w / 2) / w, (V[:, 1] + h / 2) / h]   # u along +x, v along +y (texture row 0 = +y edge, see below)
    corners = np.array([idx[0, 0], idx[0, nx - 1], idx[ny - 1, nx - 1], idx[ny - 1, 0]])
    return V, F, uv, corners, (ny, nx)


def write_obj(out, V, F, uv):
    (out / 'cloth.mtl').write_text('newmtl cloth\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nmap_Kd texture.png\n')
    with open(out / 'cloth.obj', 'w') as f:
        f.write('mtllib cloth.mtl\nusemtl cloth\n'); f.writelines(f'v {x:.6f} {y:.6f} {z:.6f}\n' for x, y, z in V)
        f.writelines(f'vt {u:.6f} {v:.6f}\n' for u, v in uv); f.write('vn 0 0 1\n')
        f.writelines(f'f {a}/{a}/1 {b}/{b}/1 {c}/{c}/1\n' for a, b, c in F + 1)


# ---------- 6. check ----------
def reproject(bgr, mask, P, V, F, w, h):
    """The mesh drawn back onto the photo. Mesh (x, y) -> plane frame (x + w/2, h/2 - y): the plane frame's y runs
    down the photo's left edge (corner 0 -> 3) while the mesh's +y is the photo's far edge."""
    pl = np.c_[V[:, 0] + w / 2, h / 2 - V[:, 1], np.zeros(len(V))]
    px = cv2.projectPoints(pl, cv2.Rodrigues(P['R'])[0], P['t'], P['K'], None)[0][:, 0]
    fill = np.zeros(mask.shape, np.uint8); cv2.fillPoly(fill, [px[f].astype(np.int32) for f in F], 1)
    iou = float((fill & mask).sum() / max((fill | mask).sum(), 1))
    vis = bgr.copy(); lw = max(1, bgr.shape[1] // 1000)
    for f in F: cv2.polylines(vis, [px[f].astype(np.int32)], True, (0, 255, 0), lw)
    return vis, iou


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--photo', required=True, type=Path); ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--side_cm', type=float, required=True, help="the cloth's LONGER side, measured with a ruler (cm)")
    ap.add_argument('--seg', default='grabcut', choices=['grabcut', 'sam', 'click', 'corners', 'mask'], help='mask: reuse OUT/mask.png from an earlier run')
    ap.add_argument('--prompt', default='cloth.', help='GroundingDINO text prompt (--seg sam)')
    ap.add_argument('--box', default=None, help='x0,y0,x1,y1: where the cloth is in the photo (--seg grabcut)')
    ap.add_argument('--corners', default=None, help='"x,y;x,y;x,y;x,y" corner pixels (--seg corners)')
    ap.add_argument('--spacing', type=float, default=0.02, help='mesh spacing (m); the simulator is tuned for 0.02')
    ap.add_argument('--f_px', type=float, default=None, help='focal length in pixels if known (default: from the vanishing points)')
    ap.add_argument('--px_per_mm', type=float, default=2.)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    bgr = imread(a.photo); H, W = bgr.shape[:2]; print(f'photo {W} x {H}')

    print(f'1. segment ({a.seg})')
    if a.seg in ('click', 'corners'):
        corners = click_corners(bgr) if a.seg == 'click' else order_corners([list(map(float, p.split(','))) for p in a.corners.split(';')])
        mask = np.zeros((H, W), np.uint8); cv2.fillPoly(mask, [corners.astype(np.int32)], 1); lines = lines_of(corners)
    else:
        if a.seg == 'mask': mask = (cv2.imdecode(np.fromfile(str(a.out / 'mask.png'), np.uint8), 0) > 127).astype(np.uint8)
        elif a.seg == 'sam': mask = segment_sam(bgr, a.prompt)
        else: mask = segment_grabcut(bgr, None if a.box is None else list(map(float, a.box.split(','))))
        imwrite(a.out / 'mask.png', mask * 255)
        print(f'  mask covers {mask.mean() * 100:.1f} % of the photo')
        print('2. fit the quadrilateral'); corners, lines = fit_quad(mask)
    ov = bgr.copy(); ov[mask > 0] = (.5 * ov[mask > 0] + [0, 0, 127]).astype(np.uint8)
    imwrite(a.out / 'mask_overlay.jpg', cv2.resize(ov, (W // 3, H // 3)))

    print('3-4. camera and size')
    P = fit_plane(corners, lines, (W, H), a.side_cm, a.f_px)
    print(f'  focal {P["f_px"]:.0f} px ({P["f_35mm"]:.1f} mm 35mm-equivalent; phones are 23-28 mm)')
    print(f'  cloth {P["width_m"] * 100:.1f} x {P["height_m"] * 100:.1f} cm (aspect {P["aspect"]:.3f}), corner reprojection {P["reproj_px"]:.1f} px')
    tex, fill = rectify(bgr, mask, P, a.px_per_mm)
    vis = bgr.copy(); cv2.polylines(vis, [corners.astype(np.int32)], True, (0, 255, 0), max(2, W // 300))
    for i, p in enumerate(corners):
        cv2.circle(vis, tuple(p.astype(int)), max(6, W // 120), (0, 0, 255), -1)
        cv2.putText(vis, str(i), tuple(p.astype(int) + W // 100), 0, W / 1000, (0, 0, 255), max(2, W // 400))
    imwrite(a.out / 'corners.jpg', cv2.resize(vis, (W // 3, H // 3)))

    print('5. grid mesh')
    w, h = P['width_m'], P['height_m']; V, F, uv, cid, shape = grid_mesh(w, h, a.spacing)
    # texture row 0 is the photo's far edge (plane y = 0) = mesh +y, so flip the image vertically for v = 0 at -y
    imwrite(a.out / 'texture.png', cv2.flip(tex, 0))
    np.savez(a.out / 'cloth_mesh.npz', vertices=V, faces=F, uv=uv, corners=cid, grid_shape=np.array(shape), width_m=w, height_m=h,
             spacing=a.spacing, source=str(a.photo.name))
    write_obj(a.out, V, F, uv)
    print(f'  {shape[1]} x {shape[0]} grid, {len(V)} vertices, {len(F)} triangles')

    print('6. check')
    rp, iou = reproject(bgr, mask, P, V, F, w, h); imwrite(a.out / 'reprojection.jpg', cv2.resize(rp, (W // 3, H // 3)))
    rep = dict(photo=str(a.photo.name), image_size=[W, H], seg=a.seg, side_cm=a.side_cm, width_cm=round(w * 100, 2), height_cm=round(h * 100, 2),
               aspect=round(P['aspect'], 4), focal_px=round(P['f_px'], 1), focal_35mm=round(P['f_35mm'], 1), corner_reproj_px=round(P['reproj_px'], 2),
               side_angle_deg=round(P['side_angle_deg'], 2), corners_px=corners.round(1).tolist(), texture_fill=round(fill, 3),
               grid=list(map(int, shape)), n_vertices=len(V), n_faces=len(F), spacing_m=a.spacing, iou_mesh_vs_mask=round(iou, 3),
               passed=bool(iou > .85 and 15 < P['f_35mm'] < 60))
    (a.out / 'mesh_report.json').write_text(json.dumps(rep, indent=2))
    print(f'  mesh vs mask IoU {iou:.3f}  ->  {"PASSED" if rep["passed"] else "CHECK the pictures"}  ({a.out / "mesh_report.json"})')


if __name__ == '__main__':
    main()
