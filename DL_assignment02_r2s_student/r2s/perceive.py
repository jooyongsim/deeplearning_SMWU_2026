"""Step 2: perception with pretrained models (given in full; run on a GPU, e.g. Colab). Frames are RGB uint8.

    box, score = detect_box(frames[q], 'white cloth.')         # Grounding DINO: text -> box
    mask = segment_frame(frames[q], box)                        # SAM 2.1: box -> mask (one image)
    masks = segment_video(frames, q, box)                       # SAM 2.1 video: propagated to every frame (optional)
    xy, vis = track_points(frames, q, px)                       # CoTracker3 online, forwards + backwards from frame q
    depth = metric_depth(frames)                                # Depth Anything V2 Metric: metres, half resolution
"""
import numpy as np
import torch

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
_cache = {}


def _load(key, fn):
    if key not in _cache: _cache[key] = fn()
    return _cache[key]


def detect_box(img, prompt='white cloth.', model='IDEA-Research/grounding-dino-tiny', thr=.3):
    """Open-vocabulary detection: the highest-scoring box (x0, y0, x1, y1) for the text prompt, and its score."""
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    proc, m = _load(model, lambda: (AutoProcessor.from_pretrained(model), AutoModelForZeroShotObjectDetection.from_pretrained(model).to(DEV).eval()))
    inp = proc(images=Image.fromarray(img), text=prompt, return_tensors='pt').to(DEV)
    with torch.no_grad(): o = m(**inp)
    det = proc.post_process_grounded_object_detection(o, inp.input_ids, threshold=thr, text_threshold=.25, target_sizes=[img.shape[:2]])[0]
    if len(det['scores']) == 0: raise RuntimeError(f'nothing found for {prompt!r}: try another prompt')
    i = int(det['scores'].argmax()); return det['boxes'][i].cpu().numpy(), float(det['scores'][i])


def segment_frame(img, box, points=None, model='facebook/sam2.1-hiera-small'):
    """SAM 2.1 on one image: a box (and optional positive points (n, 2)) -> bool mask (H, W)."""
    from PIL import Image
    from transformers import Sam2Model, Sam2Processor
    proc, m = _load(model + ':img', lambda: (Sam2Processor.from_pretrained(model), Sam2Model.from_pretrained(model).to(DEV).eval()))
    kw = dict(input_boxes=[[[float(v) for v in box]]])
    if points is not None: kw.update(input_points=[[[[float(x), float(y)] for x, y in points]]], input_labels=[[[1] * len(points)]])
    inp = proc(images=Image.fromarray(img), return_tensors='pt', **kw).to(DEV)
    with torch.no_grad(): o = m(**inp, multimask_output=False)
    return proc.post_process_masks(o.pred_masks.cpu(), inp['original_sizes'].cpu())[0][0, 0].numpy().astype(bool)


def segment_video(frames, q, box, points=None, model='facebook/sam2.1-hiera-small'):
    """SAM 2.1 video: prompt frame q with the box (and points), propagate forwards and backwards -> masks (T, H, W)."""
    from transformers import Sam2VideoModel, Sam2VideoProcessor
    proc, m = _load(model + ':video', lambda: (Sam2VideoProcessor.from_pretrained(model), Sam2VideoModel.from_pretrained(model).to(DEV).eval()))
    H, W = frames.shape[1:3]; sess = proc.init_video_session(video=list(frames), inference_device=DEV, dtype=torch.float32)
    kw = dict(input_boxes=[[[float(v) for v in box]]])
    if points is not None: kw.update(input_points=[[[[float(x), float(y)] for x, y in points]]], input_labels=[[[1] * len(points)]])
    proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=q, obj_ids=1, **kw); masks = {}
    with torch.no_grad():
        m(inference_session=sess, frame_idx=q)
        for reverse in (False, True):
            for out in m.propagate_in_video_iterator(sess, start_frame_idx=q, reverse=reverse):
                masks[out.frame_idx] = proc.post_process_masks([out.pred_masks], original_sizes=[[H, W]], binarize=True)[0][0, 0].cpu().numpy()
    return np.stack([masks[i] for i in range(len(frames))])


def _online(vid, qpts, scale):
    model = _load('cotracker3_online', lambda: torch.hub.load('facebookresearch/co-tracker', 'cotracker3_online').to(DEV).eval())
    import cv2
    small = np.stack([cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for f in vid])
    v = torch.from_numpy(small).permute(0, 3, 1, 2)[None].float().to(DEV)
    qs = torch.from_numpy(np.c_[np.zeros(len(qpts)), np.asarray(qpts) * scale]).float()[None].to(DEV)
    S = model.step; tr = vi = None
    with torch.no_grad(), torch.autocast(DEV, dtype=torch.float16, enabled=DEV == 'cuda'):
        model(video_chunk=v, is_first_step=True, queries=qs)
        for i in range(0, v.shape[1] - S, S): tr, vi = model(video_chunk=v[:, i:i + S * 2])
    tr, vi = tr[0].float().cpu().numpy() / scale, vi[0].float().cpu().numpy() > .5
    if len(tr) < len(vid):
        n = len(vid) - len(tr); tr = np.concatenate([tr, np.repeat(tr[-1:], n, 0)]); vi = np.concatenate([vi, np.repeat(vi[-1:], n, 0)])
    return tr[:len(vid)], vi[:len(vid)]


def track_points(frames, q, px, scale=.5):
    """CoTracker3 online (constant GPU memory: it reads the video in short windows), started at frame q on the points
    px (P, 2): once forwards to the end, once backwards (the reversed video) to the start. -> xy (T, P, 2), vis (T, P).
    (The offline model reads the whole video at once and runs out of memory on a T4 for ~400 frames.)"""
    fw, fwv = _online(frames[q:], px, scale)
    bw, bwv = _online(frames[:q + 1][::-1], px, scale)
    return np.concatenate([bw[::-1][:-1], fw]).astype(np.float32), np.concatenate([bwv[::-1][:-1], fwv])


def metric_depth(frames, model='depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf', every=1):
    """Depth (m) for every `every`-th frame, at half the video resolution, float16 -> (depth (T', H/2, W/2), frame indices)."""
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    proc, m = _load(model, lambda: (AutoImageProcessor.from_pretrained(model), AutoModelForDepthEstimation.from_pretrained(model).to(DEV).eval()))
    H, W = frames.shape[1:3]; out, idx = [], list(range(0, len(frames), every))
    for i in idx:
        inp = proc(images=Image.fromarray(frames[i]), return_tensors='pt').to(DEV)
        with torch.no_grad(): d = m(**inp).predicted_depth
        out.append(torch.nn.functional.interpolate(d[:, None], size=(H // 2, W // 2), mode='bilinear', align_corners=False)[0, 0].cpu().numpy().astype(np.float16))
    return np.stack(out), np.array(idx)
