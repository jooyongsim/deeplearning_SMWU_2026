#!/usr/bin/env python3
"""Pack the deliverables into submission_<student>.zip and list what is missing.

    python scripts/make_submission.py --student 20261234
Takes runs/m0-m4 (rollouts, result videos and checkpoints included), demos/<student>, il/*.py, report.pdf, slides.pptx or
slides.pdf, youtube_url.txt (the YouTube link of your presentation video, one line).
"""
import argparse, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ['runs/m0/check_env.json', 'runs/m0/scene.png', 'runs/m1/teleop.mp4', 'runs/m2/summary.csv', 'runs/m2/summary.json',
            'runs/m2/demos.png', 'runs/m3/check_il.json', 'runs/m3/ckpt.pt', 'runs/m3/train_log.csv', 'runs/m3/config.json',
            'runs/m4/zero/eval.json', 'runs/m4/bc/eval.json', 'report.pdf', 'youtube_url.txt']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--student', required=True); a = ap.parse_args()
    files = [p for p in (ROOT / 'runs').rglob('*') if p.is_file()] if (ROOT / 'runs').exists() else []
    files += sorted((ROOT / 'demos' / a.student).glob('ep_*.npz')) + sorted((ROOT / 'il').glob('*.py'))
    files += [ROOT / f for f in ('report.pdf', 'slides.pptx', 'slides.pdf', 'youtube_url.txt') if (ROOT / f).exists()]
    missing = [r for r in REQUIRED if not (ROOT / r).exists()]
    if not any((ROOT / f).exists() for f in ('slides.pptx', 'slides.pdf')): missing.append('slides.pptx (or slides.pdf)')
    n_vid = len(list((ROOT / 'runs/m4/bc/videos').glob('ep_*.mp4')))
    if n_vid < 3: missing.append(f'runs/m4/bc/videos/ep_<seed>.mp4: {n_vid} of 3 or more (scripts/render_video.py)')
    n_demo = len(list((ROOT / 'demos' / a.student).glob('ep_*.npz')))
    out = ROOT / f'submission_{a.student}.zip'
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files: z.write(f, f.relative_to(ROOT).as_posix())
    print(f'{len(files)} files ({n_demo} demos) -> {out} ({out.stat().st_size / 1e6:.1f} MB)')
    if n_demo < 30: print(f'  warning: {n_demo} demos in demos/{a.student}, 30 or more are required')
    for m in missing: print(f'  missing: {m}')
    sys.exit(1 if missing else 0)


if __name__ == '__main__':
    main()
