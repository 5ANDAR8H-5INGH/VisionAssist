"""Experimental classical hazard modules (no training needed).

These are heuristics meant to demonstrate Canny / Hough / thresholding /
morphology / contours from the lab syllabus. Replace/augment them with the
fine-tuned detector (see train_hazards.py) for better accuracy.
"""
import cv2
import numpy as np
from models import Detection


def detect_stairs(edges, shape):
    """Many near-horizontal edge lines stacked vertically in the lower part
    of the frame suggest a staircase."""
    h, w = shape[:2]
    y0 = int(h * 0.40)
    roi = edges[y0:, :]
    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=50,
                            minLineLength=int(w * 0.25), maxLineGap=15)
    if lines is None:
        return []
    lines = np.asarray(lines).reshape(-1, 4)
    ys, xs = [], []
    for x1, y1, x2, y2 in lines:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if ang < 10 or ang > 170:
            ys.append((y1 + y2) / 2.0 + y0)
            xs += [x1, x2]
    if len(ys) < 4:
        return []
    ys = sorted(ys)
    levels = [ys[0]]
    for y in ys[1:]:
        if y - levels[-1] > 10:       # a new, distinct step edge
            levels.append(y)
    if len(levels) < 4:
        return []
    box = (int(min(xs)), int(levels[0]), int(max(xs)), int(levels[-1]))
    return [Detection("stairs", 0.45, box, source="classical")]


def detect_potholes(gray, shape):
    """Dark, roughly round blobs on the ground region (Otsu + morphology)."""
    h, w = shape[:2]
    y0 = int(h * 0.60)
    roi = cv2.GaussianBlur(gray[y0:, :], (7, 7), 0)
    _, th = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out, roi_area = [], roi.shape[0] * roi.shape[1]
    for c in cnts:
        a = cv2.contourArea(c)
        if not (0.01 * roi_area < a < 0.15 * roi_area):
            continue
        per = cv2.arcLength(c, True)
        circ = 4 * np.pi * a / (per * per + 1e-6)
        if circ < 0.55:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        out.append(Detection("pothole", 0.40, (x, y + y0, x + bw, y + y0 + bh),
                             source="classical"))
    return out


def classical_hazards(edges, gray, shape):
    return detect_stairs(edges, shape) + detect_potholes(gray, shape)
