"""YOLO wrapper (Ultralytics). Optionally merges a fine-tuned hazard model."""
from models import Detection


class Detector:
    def __init__(self, cfg):
        from ultralytics import YOLO
        self.cfg = cfg
        self.coco = YOLO(cfg.weights)
        self.hazard = YOLO(cfg.hazard_weights) if cfg.hazard_weights else None
        self.allowed = set(cfg.coco_allowed)

    def _run(self, model, frame, filt=None):
        res = model.predict(frame, conf=self.cfg.conf, imgsz=self.cfg.imgsz,
                            device=self.cfg.device, verbose=False)[0]
        out = []
        for b in res.boxes:
            name = res.names[int(b.cls[0])]
            if filt is not None and name not in filt:
                continue
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            out.append(Detection(name, float(b.conf[0]), (x1, y1, x2, y2)))
        return out

    def detect(self, frame):
        dets = self._run(self.coco, frame, self.allowed)
        if self.hazard is not None:
            dets += self._run(self.hazard, frame)
        return dets
