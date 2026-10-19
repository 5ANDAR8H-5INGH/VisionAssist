# VisionAssist
Real-time obstacle detection and navigation guidance prototype (Computer Vision Lab).
Assistive prototype only - not a replacement for a white cane, guide dog or other mobility aid.

## Pipeline
camera/video -> resize -> (denoise + low-light gamma + CLAHE) -> YOLO detection
-> L/C/R zone from box centre -> MiDaS relative proximity -> IoU tracking (approaching)
-> risk level (Safe/Caution/Warning/Critical) -> overlay + spoken warning.
In parallel: Canny + morphology + Hough/contours for experimental stairs/pothole heuristics.

## Files
| File | Role |
|---|---|
| `config.py` | all thresholds/weights in one place |
| `preprocessing.py` | filters, CLAHE, gamma, Canny, morphology, contours |
| `detector.py` | YOLOv8n (COCO) + optional fine-tuned hazard model |
| `hazards.py` | classical stairs/pothole heuristics |
| `depth.py` | MiDaS relative depth (normalised 0-1, larger = closer) |
| `spatial.py` | zones + proximity category |
| `tracker.py` | IoU tracker, approaching flag |
| `risk.py` | scoring, levels, message text |
| `audio.py` | non-blocking pyttsx3 with cooldown |
| `overlay.py`, `pipeline.py`, `main.py` | display, orchestration, live demo |
| `evaluate.py` | FPS/latency comparison of pipeline variants |
| `train_hazards.py`, `hazards_dataset.yaml` | fine-tune + precision/recall/mAP |

## Run
    pip install -r requirements.txt
    python main.py                      # webcam
    python main.py --source clip.mp4 --save out.mp4
    python evaluate.py --video clip.mp4 --tag outdoor_day

Internet is needed once to download YOLO and MiDaS weights.
Keys: q quit, p preprocessing, d depth, a audio.

## Risk score
score = class_weight x proximity_weight x zone_weight (x1.25 if approaching)
Critical >= 8, Warning >= 4.5, Caution >= 2. Tune in `config.py`.

## Suggested next steps
1. Record 6-8 short clips (indoor/outdoor, static/moving, bright/low light).
2. Collect and label ~200+ images each of stairs, curbs, potholes, open drains (Roboflow/LabelImg), then run `train_hazards.py`.
3. Run `evaluate.py` per clip and add results.csv + plots to the report.
4. Tune depth thresholds in `config.py` using your own clips.
