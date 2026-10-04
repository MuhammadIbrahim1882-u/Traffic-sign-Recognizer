"""Generates a small synthetic traffic-sign dataset (12 classes).

Used automatically when the real GTSRB dataset cannot be downloaded, so the
whole project (train -> web app -> video detection) still runs end to end.
Class folders are named with the real GTSRB class ids, so every other part
of the project treats synthetic and real data the same way.

Run on its own:  python generate_synthetic_data.py --per-class 300
"""
import argparse
import glob
import math
import os

import cv2
import numpy as np

import utils

# GTSRB ids drawn here: speed 30/50/60/70, priority road, yield, stop,
# no entry, road work, turn right, turn left, keep right
SYNTHETIC_CLASS_IDS = [1, 2, 3, 4, 12, 13, 14, 17, 25, 33, 34, 38]
SPEED_TEXT = {1: "30", 2: "50", 3: "60", 4: "70"}

BASE = 192  # drawing resolution of the master sign images

# BGR colours
RED = (25, 25, 205)
WHITE = (245, 245, 245)
BLACK = (20, 20, 20)
BLUE = (200, 90, 10)
YELLOW = (0, 200, 240)


def _shrink(points, factor):
    pts = np.array(points, dtype=np.float32)
    centre = pts.mean(axis=0)
    return np.round(centre + (pts - centre) * factor).astype(np.int32)


def _put_centered(img, text, centre, target_w, color, thickness):
    font = cv2.FONT_HERSHEY_DUPLEX
    (w, _), _ = cv2.getTextSize(text, font, 1.0, 1)
    scale = target_w / float(w)
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    origin = (int(centre[0] - w / 2.0), int(centre[1] + h / 2.0))
    cv2.putText(img, text, origin, font, scale, color, thickness, cv2.LINE_8)


def _draw_sign(cid, img):
    """Draw one sign onto img (BASE x BASE, pre-filled with a background).
    No anti-aliasing here: exact edges make the two-background matting below
    precise; the later down-scaling smooths everything."""
    c = (BASE // 2, BASE // 2)
    if cid in SPEED_TEXT:
        cv2.circle(img, c, 92, RED, -1)
        cv2.circle(img, c, 68, WHITE, -1)
        _put_centered(img, SPEED_TEXT[cid], c, 88, BLACK, 7)
    elif cid == 13:  # yield: red-bordered triangle pointing down
        outer = np.array([(6, 22), (186, 22), (96, 176)], dtype=np.int32)
        cv2.fillPoly(img, [outer], RED)
        cv2.fillPoly(img, [_shrink(outer, 0.52)], WHITE)
    elif cid == 14:  # stop: red octagon with the word STOP
        angles = [math.radians(22.5 + 45 * k) for k in range(8)]
        octagon = np.array([(c[0] + 92 * math.cos(a), c[1] + 92 * math.sin(a))
                            for a in angles], dtype=np.float32)
        cv2.fillPoly(img, [np.round(octagon).astype(np.int32)], RED)
        cv2.polylines(img, [_shrink(octagon, 0.9)], True, WHITE, 5)
        _put_centered(img, "STOP", c, 116, WHITE, 6)
    elif cid == 17:  # no entry: red disc with a white bar
        cv2.circle(img, c, 92, RED, -1)
        cv2.rectangle(img, (28, 82), (164, 112), WHITE, -1)
    elif cid == 25:  # road work: red-bordered triangle pointing up + worker
        outer = np.array([(96, 6), (188, 170), (4, 170)], dtype=np.int32)
        cv2.fillPoly(img, [outer], RED)
        cv2.fillPoly(img, [_shrink(outer, 0.6)], WHITE)
        cv2.circle(img, (96, 90), 10, BLACK, -1)
        cv2.line(img, (96, 100), (96, 126), BLACK, 8)
        cv2.line(img, (96, 126), (84, 144), BLACK, 7)
        cv2.line(img, (96, 126), (108, 144), BLACK, 7)
        cv2.line(img, (96, 108), (114, 122), BLACK, 6)
    elif cid in (33, 34, 38):  # blue mandatory signs with a white arrow
        cv2.circle(img, c, 92, WHITE, -1)
        cv2.circle(img, c, 85, BLUE, -1)
        if cid == 33:
            cv2.arrowedLine(img, (38, 96), (152, 96), WHITE, 18, cv2.LINE_8, 0, 0.5)
        elif cid == 34:
            cv2.arrowedLine(img, (152, 96), (38, 96), WHITE, 18, cv2.LINE_8, 0, 0.5)
        else:
            cv2.arrowedLine(img, (56, 46), (138, 138), WHITE, 18, cv2.LINE_8, 0, 0.45)
    elif cid == 12:  # priority road: yellow diamond with white/black border
        diamond = np.array([(96, 2), (190, 96), (96, 190), (2, 96)], dtype=np.int32)
        cv2.fillPoly(img, [diamond], BLACK)
        cv2.fillPoly(img, [_shrink(diamond, 0.94)], WHITE)
        cv2.fillPoly(img, [_shrink(diamond, 0.70)], YELLOW)
    else:
        raise ValueError("No synthetic drawing for class %d" % cid)


_RENDER_CACHE = {}


def render_sign(cid, size=None):
    """Return (premultiplied BGR float32 HxWx3, alpha float32 HxWx1) for a sign.
    Two renders (black / white background) give an exact alpha channel."""
    if cid not in _RENDER_CACHE:
        black = np.zeros((BASE, BASE, 3), np.uint8)
        white = np.full((BASE, BASE, 3), 255, np.uint8)
        _draw_sign(cid, black)
        _draw_sign(cid, white)
        b = black.astype(np.float32)
        w = white.astype(np.float32)
        alpha = 1.0 - (w - b).mean(axis=2, keepdims=True) / 255.0
        _RENDER_CACHE[cid] = (b, np.clip(alpha, 0.0, 1.0))
    premult, alpha = _RENDER_CACHE[cid]
    if size and size != BASE:
        interp = cv2.INTER_AREA if size < BASE else cv2.INTER_LINEAR
        premult = cv2.resize(premult, (size, size), interpolation=interp)
        alpha = cv2.resize(alpha, (size, size), interpolation=interp).reshape(size, size, 1)
    return premult, alpha


def composite(background, premult, alpha, x, y):
    """Paste a sign with its top-left corner at (x, y). Returns float32 image."""
    out = background.astype(np.float32).copy()
    h, w = alpha.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, out.shape[1]), min(y + h, out.shape[0])
    if x1 <= x0 or y1 <= y0:
        return out
    sx, sy = x0 - x, y0 - y
    a = alpha[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]
    p = premult[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]
    out[y0:y1, x0:x1] = p + (1.0 - a) * out[y0:y1, x0:x1]
    return out


def random_background(rng, size):
    base = rng.integers(20, 235, size=3).astype(np.float32)
    delta = rng.integers(-70, 70, size=3).astype(np.float32)
    ramp = np.tile(np.linspace(0, 1, size, dtype=np.float32)[:, None], (1, size))
    if rng.integers(0, 2):
        ramp = ramp.T
    bg = base + ramp[..., None] * delta
    for _ in range(int(rng.integers(0, 4))):  # a few random blocks as clutter
        x0, y0 = rng.integers(0, size, size=2)
        x1 = min(size, x0 + int(rng.integers(6, size // 2)))
        y1 = min(size, y0 + int(rng.integers(6, size // 2)))
        bg[y0:y1, x0:x1] = rng.integers(0, 255, size=3)
    return bg


def make_sample(cid, rng, out=64):
    """One augmented sign image (BGR uint8, out x out)."""
    size = int(round(out * rng.uniform(0.78, 0.96)))
    premult, alpha = render_sign(cid, size)
    matrix = cv2.getRotationMatrix2D((size / 2.0, size / 2.0),
                                     rng.uniform(-10, 10), rng.uniform(0.95, 1.05))
    premult = cv2.warpAffine(premult, matrix, (size, size))
    alpha = cv2.warpAffine(alpha, matrix, (size, size)).reshape(size, size, 1)
    room = out - size
    x = int(np.clip(room / 2.0 + rng.integers(-3, 4), 0, room))
    y = int(np.clip(room / 2.0 + rng.integers(-3, 4), 0, room))
    img = composite(random_background(rng, out), premult, alpha, x, y)
    # lighting, blur and sensor noise
    img = img * rng.uniform(0.45, 1.25)
    img = 255.0 * np.power(np.clip(img, 0, 255) / 255.0, rng.uniform(0.8, 1.3))
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.3))
    img = img + rng.normal(0, rng.uniform(0, 9), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def generate(root, per_class=300, seed=7):
    """Create root/<class_id>/0000.png ... for every synthetic class."""
    rng = np.random.default_rng(seed)
    for cid in SYNTHETIC_CLASS_IDS:
        folder = os.path.join(root, str(cid))
        os.makedirs(folder, exist_ok=True)
        for old in glob.glob(os.path.join(folder, "*.png")):
            os.remove(old)
        for i in range(per_class):
            utils.imwrite(os.path.join(folder, "%04d.png" % i), make_sample(cid, rng))
    for stale in glob.glob(os.path.join(os.path.dirname(root), "cache_synthetic_*.npz")):
        os.remove(stale)  # the cached arrays no longer match the images
    print("Synthetic dataset ready: %d classes x %d images in %s"
          % (len(SYNTHETIC_CLASS_IDS), per_class, root))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic traffic sign dataset")
    parser.add_argument("--per-class", type=int, default=300)
    parser.add_argument("--out", default=os.path.join(utils.DATA_DIR, "synthetic"))
    args = parser.parse_args()
    generate(args.out, args.per_class)
