"""Monocular relative depth with MiDaS (scale-ambiguous by design)."""
import cv2
import numpy as np


class DepthEstimator:
    def __init__(self, cfg):
        import torch
        self.torch = torch
        self.every = max(1, cfg.depth_every_n)
        dev = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)
        # first run downloads weights via torch.hub (needs internet once)
        self.model = torch.hub.load("intel-isl/MiDaS", cfg.depth_model)
        self.model.to(self.device).eval()
        tf = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = (tf.small_transform if cfg.depth_model == "MiDaS_small"
                          else tf.dpt_transform)
        self._i = 0
        self._cache = None

    def estimate(self, frame_bgr):
        """Return HxW float map in [0,1]; larger = closer to the camera."""
        self._i += 1
        if self._cache is not None and (self._i % self.every) != 0:
            return self._cache
        torch = self.torch
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp = self.transform(rgb).to(self.device)
        with torch.no_grad():
            pred = self.model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=frame_bgr.shape[:2],
                mode="bicubic", align_corners=False).squeeze()
        d = pred.cpu().numpy().astype(np.float32)
        lo, hi = np.percentile(d, 2), np.percentile(d, 98)
        d = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)
        self._cache = d
        return d
