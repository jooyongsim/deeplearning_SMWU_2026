#!/usr/bin/env python3
"""Rope-T Arena client: connects your team's policy to the arena.

The arena runs the simulator. Each step it sends your client the 85-number observation (same layout as sim/scene.py
OBS_LAYOUT); your policy answers with a 3-number action in [-1, 1]. Nothing else crosses the wire.

From your assignment folder (the one with il/ and sim/), on a laptop:

    pip install "websockets>=12"
    python arena_client.py --url https://<arena-url> --team team3 --key <your key> --ckpt runs/m3/ckpt.pt

In a notebook (Colab / Jupyter), top-level await works:

    from arena_client import load_policy, play
    agent, info = load_policy(repo='.', ckpt='runs/m3/ckpt.pt')
    await play('https://<arena-url>', 'team3', '<your key>', agent, info)

Your policy is loaded with your own il/policy.py load_agent(), the same call il/eval_policy.py makes, so if
eval_policy.py runs your checkpoint, the arena will too. The checkpoint's SHA-256 is shown on the arena screen and
checked against your submitted checkpoint at the Q&A.
"""
import argparse, asyncio, hashlib, json, platform, sys, time
from pathlib import Path

import numpy as np


def load_policy(repo='.', ckpt='runs/m3/ckpt.pt', note=''):
    """(agent, info): agent has reset() and act(obs (85,)) -> (3,), as in il/policy.py."""
    repo, ckpt = Path(repo).resolve(), Path(ckpt).resolve()
    for p in (repo / 'il', repo / 'sim'):
        if str(p) not in sys.path: sys.path.insert(0, str(p))
    from policy import load_agent
    agent = load_agent(str(ckpt))
    params = sum(p.numel() for p in agent.model.parameters()) if hasattr(agent, 'model') else None
    info = dict(sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(), params=params, file=ckpt.name, note=note,
                platform=f'{platform.system()} py{platform.python_version()}')
    return agent, info


def _ws_url(url):
    url = url.rstrip('/')
    if url.startswith('https://'): url = 'wss://' + url[8:]
    elif url.startswith('http://'): url = 'ws://' + url[7:]
    elif not url.startswith(('ws://', 'wss://')): url = 'wss://' + url
    return url + '/ws/team'


async def play(url, team, key, agent, info=None, reconnect=True, quiet=False):
    """Connect, then play every round the arena starts until the arena says bye (or you interrupt)."""
    import websockets
    log = (lambda *a: None) if quiet else (lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True))
    ws_url, info = _ws_url(url), info or {}
    while True:
        try:
            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=60, max_size=2 ** 20, open_timeout=20) as ws:
                await ws.send(json.dumps(dict(type='hello', team=team, key=key, policy=info)))
                n_steps, t_act = 0, 0.0
                async for raw in ws:
                    m = json.loads(raw); k = m.get('type')
                    if k == 'obs':
                        t0 = time.perf_counter()
                        a = np.clip(np.asarray(agent.act(np.asarray(m['obs'], np.float32)), float).reshape(3), -1, 1)
                        t_act += time.perf_counter() - t0; n_steps += 1
                        await ws.send(json.dumps(dict(type='act', round=m['round'], t=m['t'], a=[round(float(x), 5) for x in a])))
                    elif k == 'round_start':
                        agent.reset(); n_steps, t_act = 0, 0.0
                        log(f"{m.get('label', m['round'])}: started{' (rejoined)' if m.get('rejoin') else ''}")
                    elif k == 'round_end':
                        r = m.get('result') or {}
                        if m.get('aborted'): log(f"{m['round']}: aborted by the host (does not count)")
                        else: log(f"{m['round']}: {r.get('status')} {'' if r.get('finish_s') is None else str(r['finish_s']) + ' s'}"
                                  f"  final pos {r.get('final_err', {}).get('pos')} mm  missed steps {r.get('missed')}"
                                  f"  (policy {1000 * t_act / max(n_steps, 1):.1f} ms/step)")
                    elif k == 'welcome':
                        log(m.get('msg', 'connected'), f"| policy sha256 {info.get('sha256', '?')[:12]}")
                    elif k == 'error':
                        log('arena refused the connection:', m.get('msg')); return
                    elif k == 'bye':
                        log('arena closed this connection:', m.get('msg')); return
        except (OSError, asyncio.TimeoutError, websockets.exceptions.WebSocketException) as e:
            if not reconnect: raise
            log(f'connection lost ({type(e).__name__}: {e}); retrying in 3 s')
            await asyncio.sleep(3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--url', required=True); ap.add_argument('--team', required=True); ap.add_argument('--key', required=True)
    ap.add_argument('--ckpt', default='runs/m3/ckpt.pt'); ap.add_argument('--repo', default='.')
    ap.add_argument('--note', default='', help='shown to the host, e.g. "BC chunk + PPO"')
    a = ap.parse_args()
    agent, info = load_policy(a.repo, a.ckpt, a.note)
    print(f'policy {info["file"]}  params {info["params"]}  sha256 {info["sha256"][:12]}')
    try: asyncio.run(play(a.url, a.team, a.key, agent, info))
    except KeyboardInterrupt: pass


if __name__ == '__main__':
    main()
