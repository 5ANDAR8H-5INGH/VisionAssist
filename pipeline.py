"""End-to-end VisionAssist pipeline."""
import time
from config import Config
from preprocessing import Preprocessor
from detector import Detector
from depth import DepthEstimator
from hazards import classical_hazards
from tracker import SimpleTracker
import spatial
import risk
import overlay


class VisionAssist:
    def __init__(self, cfg=None, use_pre=True, use_depth=True,
                 use_tracking=True):
        self.cfg = cfg or Config()
        self.use_pre = use_pre
        self.use_depth = use_depth
        self.use_tracking = use_tracking
        self.pre = Preprocessor(self.cfg)
        self.detector = Detector(self.cfg)
        self.depth = DepthEstimator(self.cfg) if use_depth else None
        self.tracker = SimpleTracker(self.cfg)
        self.timings = {}

    def set_depth(self, on):
        self.use_depth = on
        if on and self.depth is None:
            self.depth = DepthEstimator(self.cfg)

    def process(self, frame):
        t = {}
        t0 = time.perf_counter()
        frame = self.pre.resize(frame)

        a = time.perf_counter()
        proc = self.pre.enhance(frame) if self.use_pre else frame
        t["preprocess"] = time.perf_counter() - a

        a = time.perf_counter()
        dets = self.detector.detect(proc)
        t["detect"] = time.perf_counter() - a

        if self.cfg.use_classical_hazards:
            a = time.perf_counter()
            edges = self.pre.edges(frame)
            dets += classical_hazards(edges, self.pre.gray(frame), frame.shape)
            t["classical"] = time.perf_counter() - a

        depth_map = None
        if self.use_depth and self.depth is not None:
            a = time.perf_counter()
            depth_map = self.depth.estimate(frame)
            t["depth"] = time.perf_counter() - a

        spatial.annotate(dets, frame.shape, depth_map, self.cfg)
        if self.use_tracking:
            self.tracker.update(dets)
        risk.assess(dets, self.cfg)
        message = risk.build_message(dets, self.cfg)

        t["total"] = time.perf_counter() - t0
        self.timings = t
        return frame, dets, message

    def render(self, frame, dets, message, fps, audio_on):
        flags = {"pre": self.use_pre, "depth": self.use_depth,
                 "audio": audio_on}
        return overlay.draw(frame, dets, message, fps, flags)
