#!/usr/bin/env python3
"""Milestone 3 check: tests your dataset.py / policy.py TODOs without training (an untrained network is fine).

    python il/check_il.py --data demos/20261234
Writes runs/m3/check_il.json (deliverable of Milestone 3).
"""
import argparse, json, sys, tempfile, traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
from dataset import load_demos, split_by_episode, make_features, make_pairs, Normalizer, BCDataset
from policy import MLPPolicy, Agent, save_checkpoint, load_agent


def tests(data):
    eps = load_demos(data); assert len(eps) >= 2, f'need at least 2 valid episodes in {data}, found {len(eps)}'
    tr, va = split_by_episode(eps, .2, 0); out = {}

    def t_features():
        F = make_features(eps[0]['obs']); assert F.ndim == 2 and len(F) == eps[0]['length'], f'shape {F.shape}'
        assert np.isfinite(F).all(); return f'D = {F.shape[1]}'

    def t_pairs():
        for To, Ta in ((1, 1), (2, 4)):
            X, Y = make_pairs(tr, To, Ta); D = make_features(tr[0]['obs']).shape[1]
            assert X.shape == (sum(e['length'] for e in tr), To * D), f'X {X.shape} for To={To}'
            assert Y.shape == (len(X), Ta * 3), f'Y {Y.shape} for Ta={Ta}'
            assert X.dtype == np.float32 and Y.dtype == np.float32, 'float32 expected'
            n0 = tr[0]['length']     # the first sample of episode 2 must not see episode 1
            if len(tr) > 1 and To > 1: assert np.allclose(X[n0, :D], X[n0, D:2 * D]), 'a window crosses into the previous episode'
            assert np.allclose(Y[0, :3], tr[0]['act'][0]), 'Y[0] should start with the first action'
        return 'ok'

    # normalizer, dataset and model are tested on synthetic arrays, so each is scored apart from make_pairs
    rng = np.random.default_rng(0)
    Xs, Ys = (rng.normal(3, 2, (200, 40)) * np.r_[np.ones(39), 0]).astype(np.float32), rng.uniform(-1, 1, (200, 12)).astype(np.float32)   # last column constant

    def t_norm():
        X = Xs; n = Normalizer(); n.fit(X); Z = n(X)
        assert np.allclose(Z.mean(0), 0, atol=1e-3) and np.all((Z.std(0) < 1.01)), 'not standardised'
        assert np.allclose(n.inverse(Z), X, atol=1e-4), 'inverse does not undo'
        m = Normalizer.from_state(json.loads(json.dumps(n.state_dict(), default=lambda o: np.asarray(o).tolist()))); assert np.allclose(m(X), Z, atol=1e-5)
        return 'ok'

    def t_dataset():
        X, Y = Xs, Ys; nx, ny = Normalizer(), Normalizer(); nx.fit(X); ny.fit(Y); ds = BCDataset(X, Y, nx, ny)
        assert len(ds) == len(X); x, y = ds[0]; assert torch.is_tensor(x) and x.dtype == torch.float32 and x.shape == (X.shape[1],)
        b = next(iter(torch.utils.data.DataLoader(ds, batch_size=8))); return f'batch {tuple(b[0].shape)} {tuple(b[1].shape)}'

    def t_model():
        X, Y = Xs, Ys; m = MLPPolicy(X.shape[1], Y.shape[1]); x = torch.randn(5, X.shape[1]); y = m(x)
        assert y.shape == (5, Y.shape[1]); y.sum().backward(); assert any(p.grad is not None for p in m.parameters())
        return f'{sum(p.numel() for p in m.parameters())} parameters'

    def t_agent():
        X, Y = make_pairs(tr, 2, 4); nx, ny = Normalizer(), Normalizer(); nx.fit(X); ny.fit(Y)
        cfg = dict(model=dict(in_dim=X.shape[1], out_dim=Y.shape[1]), obs_horizon=2, act_horizon=4); m = MLPPolicy(**cfg['model'])
        with tempfile.TemporaryDirectory() as d:
            save_checkpoint(Path(d) / 'c.pt', m, cfg, nx, ny); ag = load_agent(Path(d) / 'c.pt')
        ag.reset(); acts = [np.asarray(ag.act(o)) for o in eps[0]['obs'][:10]]
        assert all(a.shape == (3,) and np.all(np.abs(a) <= 1) for a in acts), 'act must return (3,) in [-1, 1]'
        return 'ok'

    for name, f in [('features', t_features), ('pairs', t_pairs), ('normalizer', t_norm), ('dataset', t_dataset), ('model', t_model), ('agent', t_agent)]:
        try: out[name] = dict(passed=True, note=f())
        except NotImplementedError as e: out[name] = dict(passed=False, note=f'not implemented: {e}')
        except Exception as e: out[name] = dict(passed=False, note=f'{type(e).__name__}: {e}', trace=traceback.format_exc(limit=3))
        print(f'  [{"PASS" if out[name]["passed"] else "FAIL"}] {name}: {out[name]["note"]}')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', nargs='+', required=True); ap.add_argument('--out', type=Path, default=ROOT / 'runs' / 'm3')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    res = tests(a.data); passed = all(r['passed'] for r in res.values())
    (a.out / 'check_il.json').write_text(json.dumps(dict(data=a.data, tests=res, passed=passed), indent=2))
    print(f'{"PASSED" if passed else "FAILED"} -> {a.out / "check_il.json"}'); sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
