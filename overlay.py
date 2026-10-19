"""Visual overlay: boxes, zone guides, risk colours, status panel."""
import cv2

COLORS = {"Safe": (80, 200, 80), "Caution": (0, 220, 255),
          "Warning": (0, 140, 255), "Critical": (0, 0, 255)}


def draw(frame, dets, message, fps, flags):
    out = frame.copy()
    h, w = out.shape[:2]
    for x in (w // 3, 2 * w // 3):
        cv2.line(out, (x, 0), (x, h), (255, 255, 255), 1, cv2.LINE_AA)
    for d in dets:
        c = COLORS[d.risk]
        x1, y1, x2, y2 = [int(v) for v in d.box]
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        tag = f"{d.label} {d.conf:.2f} | {d.zone} | {d.proximity} | {d.risk}"
        if d.approaching:
            tag += " ^"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), c, -1)
        cv2.putText(out, tag, (x1 + 2, max(th, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    status = (f"FPS {fps:4.1f} | pre:{'on' if flags['pre'] else 'off'} "
              f"depth:{'on' if flags['depth'] else 'off'} "
              f"audio:{'on' if flags['audio'] else 'off'}")
    cv2.putText(out, status, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    if message:
        txt, _, lvl = message
        cv2.rectangle(out, (0, h - 32), (w, h), COLORS[lvl], -1)
        cv2.putText(out, txt, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 0, 0), 2, cv2.LINE_AA)
    return out
