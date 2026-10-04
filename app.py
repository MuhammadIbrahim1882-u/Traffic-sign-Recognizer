"""Web dashboard for the traffic sign recognizer.

    python app.py      then open http://127.0.0.1:5000

Two modes:
  * Single sign: the image is one sign, we classify the whole picture.
  * Road scene : we first locate signs (colour + shape), then classify each.
"""
import base64
import os

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

import utils

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB uploads

recognizer = utils.Recognizer()

MAX_SCENE_SIDE = 960
MAX_CANDIDATES = 8
SCENE_THRESHOLD = 0.80   # a scene detection must be at least this confident
UNSURE_BELOW = 0.50      # single-sign result below this is reported as unclear


def pct(value):
    return round(float(value) * 100, 1)


def error(message, status):
    return jsonify(ok=False, error=message), status


@app.errorhandler(413)
def too_large(_exc):
    return error("That image is larger than 10 MB. Try a smaller file.", 413)


@app.route("/")
def index():
    return render_template("index.html", model_ready=recognizer.ready,
                           model_error=recognizer.error)


@app.route("/api/status")
def status():
    return jsonify(ready=recognizer.ready, error=recognizer.error,
                   classes=len(recognizer.class_ids))


@app.route("/api/predict", methods=["POST"])
def predict():
    if not recognizer.ready:
        return error(recognizer.error, 503)

    upload = request.files.get("image")
    if upload is None or upload.filename == "":
        return error("No image received. Choose or drop an image first.", 400)
    data = np.frombuffer(upload.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        return error("That file could not be read as an image. Use PNG, JPG or BMP.", 400)

    mode = request.form.get("mode", "sign")
    try:
        if mode == "scene":
            return scene_response(image)
        return sign_response(image)
    except Exception as exc:  # keep the UI alive and tell the user what happened
        return error("Recognition failed: %s" % exc, 500)


def sign_response(image):
    result = recognizer.classify_batch([image], k=3)[0]
    if result["confidence"] < UNSURE_BELOW:
        decision = {"text": "Sign unclear: slow down and look again", "level": "caution"}
    else:
        decision = {"text": result["action"], "level": result["level"]}
    return jsonify(
        ok=True, mode="sign", decision=decision,
        name=result["name"], confidence=pct(result["confidence"]),
        top=[{"name": t["name"], "confidence": pct(t["confidence"])} for t in result["top"]],
    )


def scene_response(image):
    h, w = image.shape[:2]
    if max(h, w) > MAX_SCENE_SIDE:
        scale = MAX_SCENE_SIDE / float(max(h, w))
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    boxes = sorted(utils.find_sign_candidates(image), key=lambda b: b[2] * b[3], reverse=True)
    boxes = boxes[:MAX_CANDIDATES]
    detections = []
    if boxes:
        results = recognizer.classify_batch([utils.square_crop(image, b) for b in boxes])
        for box, res in zip(boxes, results):
            if res["confidence"] >= SCENE_THRESHOLD:
                detections.append({**res, "box": box})

    text, level = utils.summarize_decision(detections)
    annotated = image.copy()
    utils.draw_detections(annotated, detections)
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    data_uri = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    return jsonify(
        ok=True, mode="scene", decision={"text": text, "level": level}, annotated=data_uri,
        detections=[{"name": d["name"], "confidence": pct(d["confidence"]),
                     "action": d["action"], "level": d["level"]} for d in detections],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print("Open http://127.0.0.1:%d in your browser (Ctrl+C to stop)" % port)
    app.run(host="127.0.0.1", port=port, debug=False)
