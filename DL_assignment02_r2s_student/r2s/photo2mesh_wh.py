#!/usr/bin/env python3
"""photo2mesh with BOTH sides measured: one photo of the cloth lying flat + its width and height from a ruler -> the
same mesh files as photo2mesh.py (the simulator, the notebook and lift.py read them unchanged).

    python r2s/photo2mesh_wh.py --photo my_cloth.jpg --width_cm 27.8 --height_cm 32 --out assets/my_cloth --seg sam
    python r2s/photo2mesh_wh.py --photo my_cloth.jpg --width_cm 27.8 --height_cm 32 --out assets/my_cloth --seg click

Which side is which (the Real2Sim assignment):
    c0 ●━━━━ height (held by the two hands, 32 cm) ━━━━● c3
       ┃                                                 ┃
     width (27.8 cm)                                   width
       ┃                                                 ┃
    c1 ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━● c2
  mesh x = c0 -> c1 = --width_cm,  mesh y = c0 -> c3 = --height_cm   (the notebook's uv: u along c0->c1, v along c0->c3)

What changes compared with photo2mesh.py (--side_cm: the longer side only, the other from the photo):
  - the size is what you measured, not the photo's estimate of the aspect ratio;
  - the photo's 4 corners are matched to the measured sides: of the two ways to label them (which photo edge is
    c0 -> c1), the one whose aspect ratio in the photo agrees with width / height is used, so the photo may show the
    cloth either way round;
  - the focal length (unless --f_px) is fitted with both facts about the rectangle (right angles AND the measured
    aspect ratio), which is better constrained than the right angles alone; both estimates are printed;
  - mesh_report.json adds the photo's own aspect ratio next to the measured one: if they differ by more than a few %,
    a corner is off or the cloth was not flat (look at corners.jpg).
"""
import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from photo2mesh import (imread, imwrite, segment_sam, segment_grabcut, click_corners, order_corners, fit_quad, lines_of,
                        rectify, grid_mesh, write_obj, reproject)


def photo_shape(corners, f, c):
    """The rectangle seen through K(f): (cos of the angle between its sides, log(side 0->1 / side 0->3))."""
    K = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1.]])
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), corners.astype(np.float32))
    B = np.linalg.inv(K) @ Hm; b1, b2 = B[:, 0], B[:, 1]
    return float(b1 @ b2 / (np.linalg.norm(b1) * np.linalg.norm(b2))), float(np.log(np.linalg.norm(b1) / np.linalg.norm(b2)))


def fit_focal(corners, size, aspect):
    """f (px) that makes the photo's quad a rectangle with the measured aspect ratio (1-D search, log grid + refine)."""
    W, H = size; c = np.array([W / 2, H / 2]); la = np.log(aspect)
    def cost(f): cs, lr = photo_shape(corners, f, c); return cs ** 2 + (lr - la) ** 2
    fs = np.geomspace(.3, 4, 400) * max(W, H); i = int(np.argmin([cost(f) for f in fs]))
    lo, hi = fs[max(i - 1, 0)], fs[min(i + 1, len(fs) - 1)]
    for _ in range(40):                                                  # golden section on [lo, hi]
        a, b = hi - .618 * (hi - lo), lo + .618 * (hi - lo)
        if cost(a) < cost(b): hi = b
        else: lo = a
    f = (lo + hi) / 2; return f, cost(f)


def focal_vanishing(corners, size):
    """photo2mesh.py's estimate (right angles only): f^2 = -(v1 - c).(v2 - c); None if the view is too straight-on."""
    W, H = size; c = np.array([W / 2, H / 2]); L = lines_of(corners)
    def vp(l1, l2): v = np.cross(l1, l2); return v[:2] / v[2] if abs(v[2]) > 1e-12 else v[:2] * 1e12
    f2 = -(vp(L[0], L[2]) - c) @ (vp(L[1], L[3]) - c)
    return float(np.sqrt(f2)) if f2 > 0 and np.isfinite(f2) else None


def fit_plane_wh(corners, size, w_m, h_m, f_px=None):
    """Label the corners to match the measured sides, fix f, place the measured rectangle (solvePnP)."""
    W, H = size; c = np.array([W / 2, H / 2]); aspect = w_m / h_m
    best = None
    for r in (0, 1):                                  # r = 1: the photo edge c0 -> c1 is the other pair of sides
        q = np.roll(corners, -r, axis=0)
        f, err = (f_px, None) if f_px else fit_focal(q, size, aspect)
        cs, lr = photo_shape(q, f, c); mis = cs ** 2 + (lr - np.log(aspect)) ** 2      # right angles + measured aspect
        print(f'  labelling {r}: f {f:.0f} px, photo aspect {np.exp(lr):.3f}, corner angle {np.degrees(np.arccos(np.clip(cs, -1, 1))):.1f} deg, cost {mis:.2e}')
        if best is None or mis < best['mis']: best = dict(corners=q, f=f, mis=mis, cos=cs, photo_aspect=float(np.exp(lr)), rot=r)
    q, f = best['corners'], best['f']
    K = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1.]])
    obj = np.array([[0, 0, 0], [w_m, 0, 0], [w_m, h_m, 0], [0, h_m, 0]], float)
    ok, rvec, tvec = cv2.solvePnP(obj, q, K, None, flags=cv2.SOLVEPNP_IPPE)
    R = cv2.Rodrigues(rvec)[0]; proj = cv2.projectPoints(obj, rvec, tvec, K, None)[0][:, 0]
    return dict(K=K, R=R, t=tvec.ravel(), width_m=w_m, height_m=h_m, aspect=aspect, photo_aspect=best['photo_aspect'],
                f_px=f, f_35mm=f / np.hypot(W, H) * 43.27, reproj_px=float(np.abs(proj - q).max()),
                side_angle_deg=float(np.degrees(np.arccos(np.clip(best['cos'], -1, 1)))), corners=q, rotated=bool(best['rot']))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--photo', required=True, type=Path); ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--width_cm', type=float, required=True, help='side c0 -> c1 measured with a ruler (cm); Real2Sim: the side NOT held (27.8)')
    ap.add_argument('--height_cm', type=float, required=True, help='side c0 -> c3 measured with a ruler (cm); Real2Sim: the side held by the hands (32)')
    ap.add_argument('--seg', default='grabcut', choices=['grabcut', 'sam', 'click', 'corners', 'mask'], help='mask: reuse OUT/mask.png from an earlier run')
    ap.add_argument('--prompt', default='cloth.', help='GroundingDINO text prompt (--seg sam)')
    ap.add_argument('--box', default=None, help='x0,y0,x1,y1: where the cloth is in the photo (--seg grabcut)')
    ap.add_argument('--corners', default=None, help='"x,y;x,y;x,y;x,y" corner pixels (--seg corners)')
    ap.add_argument('--spacing', type=float, default=0.02, help='mesh spacing (m); the simulator is tuned for 0.02')
    ap.add_argument('--f_px', type=float, default=None, help='focal length in pixels if known (default: fitted to the measured rectangle)')
    ap.add_argument('--px_per_mm', type=float, default=2.)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    bgr = imread(a.photo); H, W = bgr.shape[:2]; print(f'photo {W} x {H}')

    print(f'1. segment ({a.seg})')
    if a.seg in ('click', 'corners'):
        corners = click_corners(bgr) if a.seg == 'click' else order_corners([list(map(float, p.split(','))) for p in a.corners.split(';')])
        mask = np.zeros((H, W), np.uint8); cv2.fillPoly(mask, [corners.astype(np.int32)], 1)
    else:
        if a.seg == 'mask': mask = (cv2.imdecode(np.fromfile(str(a.out / 'mask.png'), np.uint8), 0) > 127).astype(np.uint8)
        elif a.seg == 'sam': mask = segment_sam(bgr, a.prompt)
        else: mask = segment_grabcut(bgr, None if a.box is None else list(map(float, a.box.split(','))))
        imwrite(a.out / 'mask.png', mask * 255)
        print(f'  mask covers {mask.mean() * 100:.1f} % of the photo')
        print('2. fit the quadrilateral'); corners, _ = fit_quad(mask)
    ov = bgr.copy(); ov[mask > 0] = (.5 * ov[mask > 0] + [0, 0, 127]).astype(np.uint8)
    imwrite(a.out / 'mask_overlay.jpg', cv2.resize(ov, (W // 3, H // 3)))

    print('3-4. camera and size (measured)')
    w, h = a.width_cm / 100, a.height_cm / 100
    P = fit_plane_wh(corners, (W, H), w, h, a.f_px); corners = P['corners']
    fv = focal_vanishing(corners, (W, H))
    print(f'  focal {P["f_px"]:.0f} px ({P["f_35mm"]:.1f} mm 35mm-equivalent; phones are 23-28 mm)'
          + ('' if a.f_px else f'  [right angles only (photo2mesh.py): {fv:.0f} px]' if fv else '  [right angles only: uninformative]'))
    if P['rotated']: print('  the photo edge corner 0 -> 1 is the HEIGHT side: corners relabelled (c0 -> c1 = width)')
    diff = (P['photo_aspect'] / P['aspect'] - 1) * 100
    print(f'  cloth {w * 100:.1f} x {h * 100:.1f} cm (measured aspect {P["aspect"]:.3f}, in the photo {P["photo_aspect"]:.3f}: {diff:+.1f} %),'
          f' corner angle {P["side_angle_deg"]:.1f} deg, corner reprojection {P["reproj_px"]:.1f} px')
    if abs(diff) > 5: print('  WARNING: the photo disagrees with the measured sides by > 5 %: check corners.jpg (a corner off? cloth not flat?)')
    tex, fill = rectify(bgr, mask, P, a.px_per_mm)
    vis = bgr.copy(); cv2.polylines(vis, [corners.astype(np.int32)], True, (0, 255, 0), max(2, W // 300))
    for i, p in enumerate(corners):
        cv2.circle(vis, tuple(p.astype(int)), max(6, W // 120), (0, 0, 255), -1)
        cv2.putText(vis, f'c{i}', tuple(p.astype(int) + W // 100), 0, W / 1000, (0, 0, 255), max(2, W // 400))
    imwrite(a.out / 'corners.jpg', cv2.resize(vis, (W // 3, H // 3)))

    print('5. grid mesh')
    V, F, uv, cid, shape = grid_mesh(w, h, a.spacing)
    imwrite(a.out / 'texture.png', cv2.flip(tex, 0))           # texture row 0 = photo's far edge = mesh +y (as photo2mesh.py)
    np.savez(a.out / 'cloth_mesh.npz', vertices=V, faces=F, uv=uv, corners=cid, grid_shape=np.array(shape), width_m=w, height_m=h,
             spacing=a.spacing, source=str(a.photo.name))
    write_obj(a.out, V, F, uv)
    print(f'  {shape[1]} x {shape[0]} grid, {len(V)} vertices, {len(F)} triangles (actual spacing {w / (shape[1] - 1) * 100:.2f} x {h / (shape[0] - 1) * 100:.2f} cm)')

    print('6. check')
    rp, iou = reproject(bgr, mask, P, V, F, w, h); imwrite(a.out / 'reprojection.jpg', cv2.resize(rp, (W // 3, H // 3)))
    rep = dict(photo=str(a.photo.name), image_size=[W, H], seg=a.seg, width_cm=a.width_cm, height_cm=a.height_cm, measured='both sides',
               aspect=round(P['aspect'], 4), photo_aspect=round(P['photo_aspect'], 4), aspect_diff_pct=round(diff, 2), corners_relabelled=P['rotated'],
               focal_px=round(P['f_px'], 1), focal_px_right_angles_only=None if fv is None else round(fv, 1), focal_35mm=round(P['f_35mm'], 1),
               corner_reproj_px=round(P['reproj_px'], 2), side_angle_deg=round(P['side_angle_deg'], 2), corners_px=corners.round(1).tolist(),
               texture_fill=round(fill, 3), grid=list(map(int, shape)), n_vertices=len(V), n_faces=len(F), spacing_m=a.spacing,
               iou_mesh_vs_mask=round(iou, 3), passed=bool(iou > .85 and 15 < P['f_35mm'] < 60 and abs(diff) <= 5))
    (a.out / 'mesh_report.json').write_text(json.dumps(rep, indent=2))
    print(f'  mesh vs mask IoU {iou:.3f}  ->  {"PASSED" if rep["passed"] else "CHECK the pictures"}  ({a.out / "mesh_report.json"})')


if __name__ == '__main__':
    main()
