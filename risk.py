"""Risk prioritisation + spoken message construction."""
LEVELS = ["Safe", "Caution", "Warning", "Critical"]
PROX_TEXT = {"very_close": "very close", "close": "close",
             "medium": "nearby", "far": "far"}
ZONE_TEXT = {"left": "on your left", "right": "on your right",
             "center": "ahead"}


def assess(dets, cfg):
    for d in dets:
        w = cfg.class_weight.get(d.label, cfg.default_class_weight)
        s = w * cfg.proximity_weight[d.proximity] * cfg.zone_weight[d.zone]
        if d.approaching:
            s *= cfg.approaching_boost
        d.score = s
        if s >= cfg.critical_at:
            d.risk = "Critical"
        elif s >= cfg.warning_at:
            d.risk = "Warning"
        elif s >= cfg.caution_at:
            d.risk = "Caution"
        else:
            d.risk = "Safe"
    dets.sort(key=lambda d: d.score, reverse=True)
    return dets


def build_message(dets, cfg):
    """Return (text, key, level) for the single most important obstacle,
    or None if nothing reaches the speaking threshold."""
    if not dets:
        return None
    top = dets[0]
    if LEVELS.index(top.risk) < LEVELS.index(cfg.speak_min_level):
        return None
    label = top.label.replace("_", " ")
    txt = f"{label} {PROX_TEXT[top.proximity]} {ZONE_TEXT[top.zone]}"
    if top.approaching:
        txt += ", approaching"
    if top.risk == "Critical":
        txt = "Stop. " + txt
    txt = txt[0].upper() + txt[1:] + "."
    return txt, (top.label, top.zone, top.risk), top.risk
