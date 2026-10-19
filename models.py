"""Shared data structures."""
from dataclasses import dataclass


@dataclass
class Detection:
    label: str
    conf: float
    box: tuple                 # (x1, y1, x2, y2) in pixels
    source: str = "yolo"       # "yolo" | "classical"
    zone: str = "center"       # left | center | right
    proximity: str = "far"     # very_close | close | medium | far
    depth: float = 0.0         # relative inverse depth, 0 (far) .. 1 (near)
    risk: str = "Safe"         # Safe | Caution | Warning | Critical
    score: float = 0.0
    approaching: bool = False
    track_id: int = -1

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def area(self):
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)
