"""Experiment runner for the synopsis' planned comparisons.

Runs a video through several pipeline variants and logs FPS, latency and
detection counts to results.csv, then plots a bar chart.

  python evaluate.py --video indoor_lowlight.mp4 --tag indoor_lowlight
  python evaluate.py --video outdoor_day.mp4 --tag outdoor_day

Compared variants:
  detector_only   : original frames, YOLO only
  preprocessed    : classical preprocessing + YOLO
  pre+depth       : + MiDaS relative proximity + risk layer

For detection precision/recall/mAP use train_hazards.py --val-only on a
labelled dataset (see hazards_dataset.yaml). Warning accuracy needs manual
ground-truth labels per clip; log them in warnings_gt.csv and compare.
"""
import argparse
import csv
import os
import cv2
import numpy as np
from config import Config
from pipeline import VisionAssist

VARIANTS = {
    "detector_only": dict(use_pre=False, use_depth=False, use_tracking=False),
    "preprocessed": dict(use_pre=True, use_depth=False, use_tracking=False),
    "pre+depth": dict(use_pre=True, use_depth=True, use_tracking=True),
}


def run(video, name, opts, max_frames):
    cfg = Config()
    va = VisionAssist(cfg, **opts)
    cap = cv2.VideoCapture(video)
    lat, n_det, n_warn, n = [], 0, 0, 0
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        _, dets, msg = va.process(frame)
        lat.append(va.timings["total"])
        n_det += len(dets)
        n_warn += int(msg is not None)
        n += 1
    cap.release()
    lat = np.array(lat) if lat else np.array([np.nan])
    return {"variant": name, "frames": n,
            "mean_latency_ms": round(1000 * float(lat.mean()), 1),
            "p95_latency_ms": round(1000 * float(np.percentile(lat, 95)), 1),
            "fps": round(1.0 / float(lat.mean()), 2),
            "detections": n_det, "warning_frames": n_warn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tag", default="run", help="scene condition label")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    rows = []
    for name, opts in VARIANTS.items():
        print(f"running {name} ...")
        r = run(args.video, name, opts, args.max_frames)
        r["scene"] = args.tag
        rows.append(r)
        print(r)

    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.bar([r["variant"] for r in rows], [r["fps"] for r in rows])
    plt.ylabel("FPS")
    plt.title(f"Pipeline speed - {args.tag}")
    plt.tight_layout()
    plt.savefig(f"fps_{args.tag}.png", dpi=150)
    print(f"saved {args.out} and fps_{args.tag}.png")


if __name__ == "__main__":
    main()
