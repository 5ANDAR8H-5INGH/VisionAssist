"""VisionAssist live demo.

Examples:
  python main.py                       # webcam 0
  python main.py --source clip.mp4     # recorded video
  python main.py --no-depth --no-audio # faster, silent
Keys: q quit | p toggle preprocessing | d toggle depth | a toggle audio
"""
import argparse
import time
import cv2
from config import Config
from pipeline import VisionAssist
from audio import Speaker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--weights", default=None, help="YOLO weights (default yolov8n.pt)")
    ap.add_argument("--hazard-weights", default="", help="fine-tuned hazard model")
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--no-preprocess", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--save", default="", help="write annotated video to this path")
    args = ap.parse_args()

    cfg = Config()
    if args.weights:
        cfg.weights = args.weights
    cfg.hazard_weights = args.hazard_weights

    va = VisionAssist(cfg, use_pre=not args.no_preprocess,
                      use_depth=not args.no_depth)
    speaker = Speaker(enabled=not args.no_audio, cooldown=cfg.speak_cooldown)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")

    writer, fps, last = None, 0.0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame, dets, msg = va.process(frame)
        if msg:
            speaker.say(msg[0], msg[1])
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6)) if fps else 1.0 / max(now - last, 1e-6)
        last = now
        vis = va.render(frame, dets, msg, fps, speaker.enabled)

        if args.save:
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            writer.write(vis)

        cv2.imshow("VisionAssist", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("p"):
            va.use_pre = not va.use_pre
        elif k == ord("d"):
            va.set_depth(not va.use_depth)
        elif k == ord("a"):
            speaker.enabled = not speaker.enabled

    cap.release()
    if writer:
        writer.release()
    speaker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
