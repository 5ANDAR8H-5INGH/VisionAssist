"""Lightweight IoU tracker to flag approaching obstacles."""
from collections import deque


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


class SimpleTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks = {}     # id -> dict(box,label,age,hist)
        self.next_id = 1

    def update(self, dets):
        unmatched = set(self.tracks)
        pairs = []
        for i, d in enumerate(dets):
            for tid, t in self.tracks.items():
                if t["label"] == d.label:
                    s = iou(d.box, t["box"])
                    if s >= self.cfg.track_iou:
                        pairs.append((s, i, tid))
        used = set()
        for s, i, tid in sorted(pairs, reverse=True):
            if i in used or tid not in unmatched:
                continue
            used.add(i)
            unmatched.discard(tid)
            self._assign(dets[i], tid)
        for i, d in enumerate(dets):
            if i not in used:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"box": d.box, "label": d.label, "age": 0,
                                    "hist": deque(maxlen=8)}
                self._assign(d, tid)
        for tid in list(unmatched):
            self.tracks[tid]["age"] += 1
            if self.tracks[tid]["age"] > self.cfg.track_max_age:
                del self.tracks[tid]
        return dets

    def _assign(self, d, tid):
        t = self.tracks[tid]
        t["box"], t["age"] = d.box, 0
        t["hist"].append((d.area, d.depth))
        d.track_id = tid
        h = t["hist"]
        if len(h) >= 4:
            a0, d0 = h[0]
            a1, d1 = h[-1]
            grew = a0 > 0 and (a1 / a0) >= self.cfg.approach_area_ratio
            nearer = (d1 - d0) >= self.cfg.approach_depth_delta
            d.approaching = bool(grew or nearer)
