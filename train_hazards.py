"""Fine-tune / validate YOLO on the custom navigation-hazard dataset.

  python train_hazards.py --epochs 50
  python train_hazards.py --val-only --weights runs/detect/train/weights/best.pt
Prints precision, recall, mAP50 and mAP50-95.
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hazards_dataset.yaml")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--val-only", action="store_true")
    a = ap.parse_args()

    model = YOLO(a.weights)
    if not a.val_only:
        model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz)
    m = model.val(data=a.data, imgsz=a.imgsz)
    print(f"precision={m.box.mp:.3f} recall={m.box.mr:.3f} "
          f"mAP50={m.box.map50:.3f} mAP50-95={m.box.map:.3f}")


if __name__ == "__main__":
    main()
