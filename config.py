"""Central configuration for VisionAssist."""
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- input ---
    frame_width: int = 640

    # --- preprocessing ---
    denoise: str = "bilateral"        # gaussian | median | bilateral | none
    use_clahe: bool = True
    low_light_mean: float = 70.0      # mean gray below this -> gamma brighten
    low_light_gamma: float = 0.6

    # --- detection ---
    weights: str = "yolov8n.pt"       # lightweight COCO model
    hazard_weights: str = ""          # optional fine-tuned model (stairs, curb, ...)
    conf: float = 0.40
    imgsz: int = 640
    device: str = None                # None = auto
    coco_allowed: tuple = (
        "person", "bicycle", "car", "motorcycle", "bus", "truck",
        "chair", "bench", "dining table", "couch", "potted plant",
        "traffic light", "stop sign", "fire hydrant", "dog", "cat",
        "suitcase", "backpack", "bed", "tv", "refrigerator",
    )

    # --- depth (MiDaS, relative) ---
    depth_model: str = "MiDaS_small"  # MiDaS_small | DPT_Hybrid | DPT_Large
    depth_every_n: int = 3            # run depth every N frames (speed)
    # thresholds on normalised inverse depth (higher = closer)
    depth_very_close: float = 0.75
    depth_close: float = 0.55
    depth_medium: float = 0.35
    # fallback when depth is off: box height / frame height
    box_very_close: float = 0.70
    box_close: float = 0.45
    box_medium: float = 0.20

    # --- risk ---
    class_weight: dict = field(default_factory=lambda: {
        "car": 3.0, "bus": 3.0, "truck": 3.0, "motorcycle": 3.0,
        "stairs": 3.0, "pothole": 3.0, "open_drain": 3.0, "curb": 2.5,
        "bicycle": 2.5, "person": 2.0, "dog": 2.0,
        "chair": 1.5, "bench": 1.5, "dining table": 1.5, "couch": 1.5,
        "fire hydrant": 1.5,
    })
    default_class_weight: float = 1.0
    proximity_weight: dict = field(default_factory=lambda: {
        "very_close": 4, "close": 3, "medium": 2, "far": 1})
    zone_weight: dict = field(default_factory=lambda: {
        "center": 1.0, "left": 0.6, "right": 0.6})
    approaching_boost: float = 1.25
    # score thresholds
    caution_at: float = 2.0
    warning_at: float = 4.5
    critical_at: float = 8.0

    # --- tracking ---
    track_iou: float = 0.3
    track_max_age: int = 10
    approach_area_ratio: float = 1.12
    approach_depth_delta: float = 0.08

    # --- audio ---
    speak_min_level: str = "Warning"  # Caution | Warning | Critical
    speak_cooldown: float = 3.0

    # --- classical hazard modules (experimental) ---
    use_classical_hazards: bool = True
