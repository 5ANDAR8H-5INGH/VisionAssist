"""Classical image-processing stages: filtering, contrast, edges, contours."""
import cv2
import numpy as np


class Preprocessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def resize(self, frame):
        h, w = frame.shape[:2]
        if w == self.cfg.frame_width:
            return frame
        s = self.cfg.frame_width / w
        return cv2.resize(frame, (self.cfg.frame_width, int(h * s)),
                          interpolation=cv2.INTER_AREA)

    def denoise(self, frame):
        mode = self.cfg.denoise
        if mode == "gaussian":
            return cv2.GaussianBlur(frame, (5, 5), 0)
        if mode == "median":
            return cv2.medianBlur(frame, 5)
        if mode == "bilateral":
            return cv2.bilateralFilter(frame, 7, 50, 50)
        return frame

    @staticmethod
    def _gamma(frame, gamma):
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)],
                       dtype=np.uint8)
        return cv2.LUT(frame, lut)

    def enhance(self, frame):
        """Denoise + low-light gamma + CLAHE on the L channel.
        Returns a 3-channel BGR image (YOLO expects 3 channels)."""
        out = self.denoise(frame)
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        if gray.mean() < self.cfg.low_light_mean:
            out = self._gamma(out, self.cfg.low_light_gamma)
        if self.cfg.use_clahe:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        return out

    @staticmethod
    def gray(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def edges(self, frame):
        """Auto-threshold Canny + morphological closing."""
        g = cv2.GaussianBlur(self.gray(frame), (5, 5), 0)
        med = float(np.median(g))
        lo = int(max(0, 0.67 * med))
        hi = int(min(255, 1.33 * med))
        e = cv2.Canny(g, lo, hi)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(e, cv2.MORPH_CLOSE, k)

    @staticmethod
    def contours(edge_img, min_area=200):
        cnts, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in cnts if cv2.contourArea(c) >= min_area]
