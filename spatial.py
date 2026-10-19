"""Spatial reasoning: left/center/right zone + relative proximity."""
import numpy as np


def zone_of(cx, width):
    if cx < width / 3.0:
        return "left"
    if cx > 2.0 * width / 3.0:
        return "right"
    return "center"


def proximity_from_depth(depth_map, box, cfg):
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = depth_map.shape
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0, "far"
    # lower half of the box ~ where the object touches the ground
    ymid = y1 + (y2 - y1) // 2
    patch = depth_map[ymid:y2, x1:x2]
    if patch.size == 0:
        patch = depth_map[y1:y2, x1:x2]
    d = float(np.median(patch))
    if d >= cfg.depth_very_close:
        return d, "very_close"
    if d >= cfg.depth_close:
        return d, "close"
    if d >= cfg.depth_medium:
        return d, "medium"
    return d, "far"


def proximity_from_box(box, frame_h, cfg):
    r = (box[3] - box[1]) / float(frame_h)
    if r >= cfg.box_very_close:
        return r, "very_close"
    if r >= cfg.box_close:
        return r, "close"
    if r >= cfg.box_medium:
        return r, "medium"
    return r, "far"


def annotate(dets, frame_shape, depth_map, cfg):
    h, w = frame_shape[:2]
    for d in dets:
        d.zone = zone_of(d.center[0], w)
        if depth_map is not None:
            d.depth, d.proximity = proximity_from_depth(depth_map, d.box, cfg)
        else:
            d.depth, d.proximity = proximity_from_box(d.box, h, cfg)
    return dets
