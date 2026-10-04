"""Creates test material in samples/ (no download needed):

  sign_<id>_<name>.png   close-ups of single signs (for predict.py / web app)
  scene_*.jpg            street scenes with a sign by the road
  demo_drive.mp4         a short "drive" where signs approach the camera

Run:  python make_samples.py
"""
import os

import cv2
import numpy as np

import generate_synthetic_data as gen
import utils

W, H = 640, 360
FPS = 24


def slug(text):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def road_background():
    """Simple street: pale sky, grass, grey road with lane dashes."""
    img = np.zeros((H, W, 3), np.uint8)
    horizon = 150
    for row in range(horizon):  # low-saturation sky so it is not mistaken for a blue sign
        t = row / float(horizon)
        img[row, :] = (int(215 - 25 * t), int(205 - 20 * t), int(195 - 15 * t))
    img[horizon:, :] = (70, 135, 75)
    road = np.array([(285, horizon), (355, horizon), (W + 260, H), (-260, H)], np.int32)
    cv2.fillPoly(img, [road], (92, 92, 95))
    for k in range(6):  # lane dashes get longer toward the camera
        y0 = horizon + int((H - horizon) * (k / 6.0) ** 1.6)
        y1 = horizon + int((H - horizon) * ((k + 0.5) / 6.0) ** 1.6)
        w0 = 1 + int(6 * (k / 6.0))
        cv2.line(img, (320, y0), (320, y1), (235, 235, 235), w0)
    return img


def scene_frame(cid, t, background):
    """Frame with sign `cid` at approach progress t in 0..1 (None = no sign)."""
    frame = background.copy()
    if cid is None:
        return frame
    size = int(36 + 114 * t ** 1.5)
    cx = int(430 + 110 * t)
    cy = int(125 + 50 * t)
    top = cy + size // 2
    cv2.rectangle(frame, (cx - 3, top), (cx + 3, min(H, top + int(size * 1.1))), (120, 120, 125), -1)
    premult, alpha = gen.render_sign(cid, size)
    out = gen.composite(frame, premult, alpha, cx - size // 2, cy - size // 2)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    os.makedirs(utils.SAMPLES_DIR, exist_ok=True)
    rng = np.random.default_rng(3)

    # 1) single-sign close-ups (augmented like the training data, larger)
    for cid in gen.SYNTHETIC_CLASS_IDS:
        img = gen.make_sample(cid, rng, out=128)
        name = "sign_%d_%s.png" % (cid, slug(utils.class_name(cid)))
        utils.imwrite(os.path.join(utils.SAMPLES_DIR, name), img)

    # 2) street scenes
    background = road_background()
    for cid, label in ((14, "stop"), (2, "speed50"), (33, "turn_right"), (13, "yield")):
        utils.imwrite(os.path.join(utils.SAMPLES_DIR, "scene_%s.jpg" % label),
                      scene_frame(cid, 0.8, background))

    # 3) demo video: four signs approach one after another
    path = os.path.join(utils.SAMPLES_DIR, "demo_drive.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        print("Could not create the demo video (video codec missing). Images were still created.")
    else:
        for cid in (14, 2, 33, 13):
            for i in range(72):
                writer.write(scene_frame(cid, i / 71.0, background))
            for _ in range(10):  # empty road between signs
                writer.write(scene_frame(None, 0, background))
        writer.release()
    print("Samples written to", utils.SAMPLES_DIR)


if __name__ == "__main__":
    main()
