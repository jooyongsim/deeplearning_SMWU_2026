#!/usr/bin/env python3
"""Pack the deliverables into one zip: submission_<team>.zip.

    python scripts/make_submission.py --student T03
Includes runs/m0-m4 (reports, logs, curves, checkpoints, rollouts' eval.json, videos), your cloth mesh folder
(assets/<team>_cloth), demos/<team>, rl/*.py, report.pdf, slides.pptx / slides.pdf, youtube_url.txt. Lists what is missing.
"""
import argparse, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--student', required=True); ap.add_argument('--mesh', type=Path, help='your cloth mesh folder (default: assets/<team>_cloth)')
    a = ap.parse_args(); mesh = a.mesh or ROOT / 'assets' / f'{a.student}_cloth'
    want = {
        'M0': ['runs/m0/check_env.json', 'runs/m0/scene.png'],
        'M1': [f'{mesh.relative_to(ROOT).as_posix()}/{f}' for f in ('cloth_mesh.npz', 'texture.png', 'mesh_report.json', 'corners.jpg', 'reprojection.jpg', 'photo.jpg')]
              + ['runs/m1/check_env.json', 'runs/m1/scene.png'],
        'M2': ['runs/m2/teleop.mp4'],
        'M3': ['rl/features.py', 'rl/reward.py', 'runs/m3/check_rl.json', 'runs/m3/ppo/ckpt.zip', 'runs/m3/ppo/train_log.csv', 'runs/m3/ppo/curve.png', 'runs/m3/ppo/config.json'],
        'M4': ['runs/m4/random/eval.json', 'runs/m4/ppo/eval.json', 'runs/m4/ppo_mycloth/eval.json'],
        'M5': ['report.pdf', 'youtube_url.txt'],
    }
    files, missing = [], []
    for m, fs in want.items():
        for f in fs: (files if (ROOT / f).exists() else missing).append((m, f))
    extra = [p for p in (ROOT / 'demos' / a.student).glob('ep_*.npz')] + list((ROOT / 'runs' / 'm4').glob('*/videos/*.mp4')) \
        + [p for p in (ROOT / 'runs').glob('m2/*.mp4')] + [p for p in (ROOT / 'runs' / 'm3').glob('**/*.json')] + [p for p in ROOT.glob('slides.*')]
    if not any(ROOT.glob('slides.*')): missing.append(('M5', 'slides.pptx or slides.pdf'))
    if not (ROOT / 'demos' / a.student).exists() or not any((ROOT / 'demos' / a.student).glob('ep_*.npz')): missing.append(('M2', f'demos/{a.student}/ep_*.npz'))
    out = ROOT / f'submission_{a.student}.zip'; n = 0
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted({ROOT / f for _, f in files} | set(extra)):
            z.write(p, p.relative_to(ROOT).as_posix()); n += 1
    print(f'{n} files -> {out} ({out.stat().st_size / 1e6:.1f} MB)')
    if missing:
        print('missing:'); [print(f'  [{m}] {f}') for m, f in missing]


if __name__ == '__main__':
    main()
