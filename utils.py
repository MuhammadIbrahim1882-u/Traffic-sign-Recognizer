"""Shared helpers for the Traffic Sign Recognizer.

Contains:
  * the 43 GTSRB class names and the "driving action" the car should take
  * safe image read/write helpers (work with non-ASCII Windows paths)
  * Recognizer: loads the trained CNN once and classifies sign crops
  * find_sign_candidates: OpenCV colour/shape segmentation to locate signs
  * drawing helpers for bounding boxes and the decision banner

TensorFlow is imported lazily, so scripts that do not need the model
(data generation, sample creation) start fast and never fail on TF.
"""
import json
import os
import threading

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
H5_PATH = os.path.join(MODEL_DIR, "traffic_sign_model.h5")
KERAS_PATH = os.path.join(MODEL_DIR, "traffic_sign_model.keras")
BEST_PATH = os.path.join(MODEL_DIR, "best_checkpoint.keras")
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.json")
IMG_SIZE = 32  # the CNN works on 32x32 RGB crops

FONT = cv2.FONT_HERSHEY_SIMPLEX

# --------------------------------------------------------------------------
# The 43 GTSRB classes
# --------------------------------------------------------------------------
CLASS_NAMES = [
    "Speed limit (20 km/h)",                                  # 0
    "Speed limit (30 km/h)",                                  # 1
    "Speed limit (50 km/h)",                                  # 2
    "Speed limit (60 km/h)",                                  # 3
    "Speed limit (70 km/h)",                                  # 4
    "Speed limit (80 km/h)",                                  # 5
    "End of speed limit (80 km/h)",                           # 6
    "Speed limit (100 km/h)",                                 # 7
    "Speed limit (120 km/h)",                                 # 8
    "No passing",                                             # 9
    "No passing for vehicles over 3.5 tons",                  # 10
    "Right-of-way at next intersection",                      # 11
    "Priority road",                                          # 12
    "Yield",                                                  # 13
    "Stop",                                                   # 14
    "No vehicles",                                            # 15
    "Vehicles over 3.5 tons prohibited",                      # 16
    "No entry",                                               # 17
    "General caution",                                        # 18
    "Dangerous curve to the left",                            # 19
    "Dangerous curve to the right",                           # 20
    "Double curve",                                           # 21
    "Bumpy road",                                             # 22
    "Slippery road",                                          # 23
    "Road narrows on the right",                              # 24
    "Road work",                                              # 25
    "Traffic signals",                                        # 26
    "Pedestrians",                                            # 27
    "Children crossing",                                      # 28
    "Bicycles crossing",                                      # 29
    "Beware of ice/snow",                                     # 30
    "Wild animals crossing",                                  # 31
    "End of all speed and passing limits",                    # 32
    "Turn right ahead",                                       # 33
    "Turn left ahead",                                        # 34
    "Ahead only",                                             # 35
    "Go straight or right",                                   # 36
    "Go straight or left",                                    # 37
    "Keep right",                                             # 38
    "Keep left",                                              # 39
    "Roundabout mandatory",                                   # 40
    "End of no passing",                                      # 41
    "End of no passing by vehicles over 3.5 tons",            # 42
]

# What the self-driving car should do. Level drives colours in the UI:
#   stop    -> halt / do not enter
#   caution -> slow down, be alert
#   info    -> a direction or instruction to follow
#   ok      -> a restriction ends, normal driving
ACTIONS = {
    0: ("Reduce speed to 20 km/h", "caution"),
    1: ("Reduce speed to 30 km/h", "caution"),
    2: ("Reduce speed to 50 km/h", "caution"),
    3: ("Reduce speed to 60 km/h", "caution"),
    4: ("Reduce speed to 70 km/h", "caution"),
    5: ("Reduce speed to 80 km/h", "caution"),
    6: ("Speed limit ends: resume normal road speed", "ok"),
    7: ("Limit speed to 100 km/h", "caution"),
    8: ("Limit speed to 120 km/h", "caution"),
    9: ("No overtaking: stay in your lane", "caution"),
    10: ("No overtaking for heavy vehicles: stay in lane", "caution"),
    11: ("Junction ahead where you have right-of-way: proceed with care", "caution"),
    12: ("Priority road: keep speed, you have right-of-way", "ok"),
    13: ("Give way: slow down and yield to traffic", "caution"),
    14: ("STOP the vehicle completely, go only when clear", "stop"),
    15: ("Road closed to all vehicles: do not enter, reroute", "stop"),
    16: ("Heavy vehicles prohibited: reroute if this applies", "caution"),
    17: ("No entry: do not enter, reroute", "stop"),
    18: ("General caution: slow down and stay alert", "caution"),
    19: ("Curve to the left ahead: slow down", "caution"),
    20: ("Curve to the right ahead: slow down", "caution"),
    21: ("Double curve ahead: slow down", "caution"),
    22: ("Bumpy road: reduce speed", "caution"),
    23: ("Slippery road: reduce speed, avoid hard braking", "caution"),
    24: ("Road narrows on the right: slow down, move left", "caution"),
    25: ("Road work ahead: slow down, watch for workers", "caution"),
    26: ("Traffic signals ahead: prepare to stop", "caution"),
    27: ("Pedestrian crossing: slow down, be ready to stop", "caution"),
    28: ("Children crossing: slow down, be ready to stop", "caution"),
    29: ("Cyclists crossing: slow down, give them space", "caution"),
    30: ("Ice or snow risk: slow down, keep distance", "caution"),
    31: ("Wild animals may cross: slow down", "caution"),
    32: ("All restrictions end: resume normal driving", "ok"),
    33: ("Turn right ahead", "info"),
    34: ("Turn left ahead", "info"),
    35: ("Proceed straight ahead only", "info"),
    36: ("Go straight or turn right", "info"),
    37: ("Go straight or turn left", "info"),
    38: ("Keep right", "info"),
    39: ("Keep left", "info"),
    40: ("Roundabout: enter and give way to traffic inside", "caution"),
    41: ("No-passing zone ends: overtaking allowed", "ok"),
    42: ("No-passing zone for heavy vehicles ends", "ok"),
}

LEVEL_PRIORITY = {"stop": 3, "caution": 2, "info": 1, "ok": 0}
# BGR colours used when drawing with OpenCV
LEVEL_COLORS = {
    "stop": (60, 60, 225),
    "caution": (0, 150, 255),
    "info": (215, 150, 40),
    "ok": (110, 190, 60),
}


def class_name(class_id):
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return "Class %d" % class_id


def action_for(class_id):
    """Return (driving action text, level) for a GTSRB class id."""
    return ACTIONS.get(class_id, ("Unknown sign: slow down", "caution"))


# --------------------------------------------------------------------------
# Image IO (np.fromfile / tofile keeps non-ASCII Windows paths working)
# --------------------------------------------------------------------------
def imread(path):
    """Read an image as BGR uint8, or None when it cannot be read."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return bool(ok)


def bgr_to_input(crop_bgr):
    """BGR crop of any size -> RGB float32 32x32 (0..255), the model input."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return rgb.astype("float32")


# --------------------------------------------------------------------------
# Recognizer
# --------------------------------------------------------------------------
class Recognizer:
    """Loads the trained model once and classifies BGR sign crops."""

    def __init__(self):
        self.model = None
        self.error = None
        self.class_ids = list(range(len(CLASS_NAMES)))
        self._lock = threading.Lock()
        self._load()

    @property
    def ready(self):
        return self.model is not None

    def _load(self):
        candidates = [p for p in (H5_PATH, KERAS_PATH) if os.path.exists(p)]
        if not candidates:
            self.error = ("The model is not trained yet. Run 'python train.py' "
                          "in the project folder, then restart this program.")
            return
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        try:
            import tensorflow as tf
        except Exception as exc:  # TensorFlow missing or broken
            self.error = "TensorFlow could not be imported: %s" % exc
            return
        last_error = None
        for path in candidates:
            try:
                self.model = tf.keras.models.load_model(path, compile=False)
                break
            except Exception as exc:
                last_error = exc
        if self.model is None:
            self.error = "The saved model could not be loaded: %s" % last_error
            return
        if os.path.exists(CLASSES_PATH):
            try:
                with open(CLASSES_PATH, "r", encoding="utf-8") as fh:
                    self.class_ids = [int(c) for c in json.load(fh)["class_ids"]]
            except (OSError, ValueError, KeyError):
                pass
        out_dim = int(self.model.output_shape[-1])
        if out_dim != len(self.class_ids):
            self.class_ids = list(range(out_dim))

    def classify_batch(self, crops_bgr, k=3):
        """Classify a list of BGR crops. Returns one result dict per crop."""
        if not self.ready:
            raise RuntimeError(self.error)
        if not crops_bgr:
            return []
        batch = np.stack([bgr_to_input(c) for c in crops_bgr]).astype("float32")
        with self._lock:
            probs = np.asarray(self.model(batch, training=False))
        results = []
        for p in probs:
            order = np.argsort(p)[::-1][:k]
            top = []
            for i in order:
                cid = self.class_ids[int(i)]
                top.append({"class_id": cid, "name": class_name(cid),
                            "confidence": float(p[int(i)])})
            best = top[0]
            text, level = action_for(best["class_id"])
            results.append({**best, "action": text, "level": level, "top": top})
        return results


# --------------------------------------------------------------------------
# Sign localisation (colour + shape segmentation)
# --------------------------------------------------------------------------
def _intersection_over_min(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(max(1, min(aw * ah, bw * bh)))


def _suppress_overlaps(boxes, thr=0.6):
    kept = []
    for box in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
        if all(_intersection_over_min(box, k) < thr for k in kept):
            kept.append(box)
    return kept


def find_sign_candidates(frame_bgr, min_area_ratio=0.0004, max_area_ratio=0.30):
    """Find regions that look like traffic signs: red, blue or yellow blobs
    with a roughly square bounding box. Returns a list of (x, y, w, h)."""
    h, w = frame_bgr.shape[:2]
    frame_area = float(h * w)
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame_bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)

    # (mask, padding as a fraction of the box). Padding restores the part of
    # the sign outside the coloured region (e.g. the border of a yellow sign).
    red = cv2.inRange(hsv, (0, 100, 70), (10, 255, 255)) | \
        cv2.inRange(hsv, (165, 100, 70), (180, 255, 255))
    blue = cv2.inRange(hsv, (100, 130, 50), (130, 255, 255))
    yellow = cv2.inRange(hsv, (18, 110, 110), (35, 255, 255))
    layers = [(red, 0.02), (blue, 0.05), (yellow, 0.22)]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boxes = []
    for mask, pad in layers:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            box_area = bw * bh
            if box_area < min_area_ratio * frame_area or box_area > max_area_ratio * frame_area:
                continue
            if min(bw, bh) < 12:
                continue
            aspect = bw / float(bh)
            if aspect < 0.6 or aspect > 1.6:
                continue
            if cv2.contourArea(cnt) / float(box_area) < 0.30:
                continue
            px, py = int(bw * pad), int(bh * pad)
            x0, y0 = max(0, x - px), max(0, y - py)
            x1, y1 = min(w, x + bw + px), min(h, y + bh + py)
            boxes.append((x0, y0, x1 - x0, y1 - y0))
    return _suppress_overlaps(boxes)


def square_crop(frame_bgr, box, pad=0.06):
    """Crop a square around the box (with a little padding), clipped to the frame."""
    h, w = frame_bgr.shape[:2]
    x, y, bw, bh = box
    side = int(max(bw, bh) * (1 + 2 * pad))
    cx, cy = x + bw / 2.0, y + bh / 2.0
    x0 = int(max(0, min(w - side, cx - side / 2.0))) if side < w else 0
    y0 = int(max(0, min(h - side, cy - side / 2.0))) if side < h else 0
    return frame_bgr[y0:y0 + side, x0:x0 + side]


# --------------------------------------------------------------------------
# Decision logic and drawing
# --------------------------------------------------------------------------
def summarize_decision(detections):
    """Pick the most important detection: stop > caution > info > ok."""
    if not detections:
        return "Road clear: no traffic sign detected", "ok"
    best = max(detections, key=lambda d: (LEVEL_PRIORITY[d["level"]], d["confidence"]))
    return best["action"], best["level"]


def draw_detections(frame, detections):
    """Draw boxes + labels. Each detection needs box, name, confidence, level."""
    frame_w = frame.shape[1]
    for d in detections:
        x, y, w, h = d["box"]
        color = LEVEL_COLORS[d["level"]]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = "%s %d%%" % (d["name"], round(d["confidence"] * 100))
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        lx = max(0, min(x, frame_w - tw - 8))  # keep the label inside the picture
        ty = y - 8 if y >= th + 14 else y + h + th + 8
        cv2.rectangle(frame, (lx, ty - th - 4), (lx + tw + 6, ty + 4), color, -1)
        cv2.putText(frame, label, (lx + 3, ty), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_hud(frame, text, level):
    """Decision banner across the top of the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), LEVEL_COLORS[level], -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    label = "Decision: " + text
    scale = 0.65
    (tw, _), _ = cv2.getTextSize(label, FONT, scale, 2)
    if tw > w - 24:
        scale = max(0.35, scale * (w - 24) / float(tw))
    cv2.putText(frame, label, (12, 27), FONT, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return frame
