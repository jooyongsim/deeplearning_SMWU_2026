"""How folded the cloth is (no MuJoCo; the simulator, the evaluation, the RL reward and the grader share it).

The task: fold the cloth twice along its diagonals ("napkin fold"): one corner onto the opposite corner (a triangle),
then one of the two remaining corners onto the other (a smaller triangle, 4 layers at the tip). Either diagonal may
go first and either corner of a pair may be the one moved; only the end state counts.

    gap_a   distance between corners c0 and c2 (one diagonal pair), mm, in the table plane
    gap_b   distance between corners c1 and c3 (the other pair), mm
    cover   area the cloth covers seen from above / its flat area (1 = spread out, ~0.5 after one diagonal
            fold, ~0.25 after two)
    success max(gap_a, gap_b) <= GAP_MM  and  cover <= COVER   (both folds done, and neatly: not just bunched)
    folds   how many corner pairs are closed (gap <= GAP_MM): 0, 1, 2

THRESH (2026-11-18): the scripted expert (pick-and-place at the true corners) ends with a largest corner gap of 48 mm
(median) on the 50 eval seeds, 88 % within 70 mm; covered area 0.12-0.27. 70 mm is 1/6 of the cloth diagonal.
"""
import cv2
import numpy as np

THRESH = dict(gap=70., cover=0.35)      # mm, fraction (the scripted expert: 88 % of the 50 eval seeds)
RES = 0.002                              # coverage raster (m / pixel)


def silhouette(xy, faces, origin, n):
    img = np.zeros((n, n), np.uint8)
    P = np.round((xy - origin) / RES).astype(np.int32)
    cv2.fillPoly(img, list(P[faces]), 1)
    return img


def coverage(xy, faces, flat_area):
    """Covered area seen from above / flat area. xy (N, 2) vertex positions, faces (M, 3)."""
    lo = xy.min(0) - .01; n = int(np.ceil((xy.max(0) + .01 - lo).max() / RES)) + 2
    return float(silhouette(xy, faces, lo, n).sum() * RES * RES / flat_area)


def measure(xy, faces, corners, flat_area, th=None):
    """{'gap_a', 'gap_b' (mm), 'cover', 'folds', 'success'} for vertex positions xy (N, 2)."""
    th = th or THRESH; c = corners
    ga = float(np.linalg.norm(xy[c[0]] - xy[c[2]]) * 1000); gb = float(np.linalg.norm(xy[c[1]] - xy[c[3]]) * 1000)
    cov = coverage(xy, faces, flat_area)
    folds = int(ga <= th['gap']) + int(gb <= th['gap'])
    return dict(gap_a=ga, gap_b=gb, cover=cov, folds=folds, success=bool(folds == 2 and cov <= th['cover']))
