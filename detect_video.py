"""Real-time traffic sign recognition (the "car camera" demo).

    python detect_video.py                                  # webcam 0
    python detect_video.py --source samples\\demo_drive.mp4  # video file
    python detect_video.py --source samples\\scene_stop.jpg  # single street image
    python detect_video.py --source samples\\demo_drive.mp4 --output result.mp4

How it works for every frame:
  1. OpenCV finds red / blue / yellow sign-shaped regions (colour + shape).
  2. Each region is cropped and classified by the CNN.
  3. Boxes, names and confidences are drawn, and the banner at the top shows
     the car's decision (stop > caution > info > ok).

Press 'q' in the video window to quit.
"""
import argparse
import os
import sys
import time

import cv2

import utils

MAX_WIDTH = 960
MAX_CANDIDATES = 8
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".ppm")


def process_frame(frame, recognizer, threshold):
    """Detect + classify signs in a frame. Draws on the frame in place.
    Returns (frame, detections, (decision_text, level))."""
    boxes = utils.find_sign_candidates(frame)
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:MAX_CANDIDATES]
    detections = []
    if boxes:
        crops = [utils.square_crop(frame, b) for b in boxes]
        results = recognizer.classify_batch(crops)
        for box, res in zip(boxes, results):
            if res["confidence"] >= threshold:
                detections.append({**res, "box": box})
    decision = utils.summarize_decision(detections)
    utils.draw_detections(frame, detections)
    utils.draw_hud(frame, decision[0], decision[1])
    return frame, detections, decision


def resize_if_needed(frame):
    h, w = frame.shape[:2]
    if w > MAX_WIDTH:
        scale = MAX_WIDTH / float(w)
        frame = cv2.resize(frame, (MAX_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame


def run_image(path, recognizer, threshold, output):
    frame = utils.imread(path)
    if frame is None:
        sys.exit("Could not read image: %s" % path)
    frame, detections, decision = process_frame(resize_if_needed(frame), recognizer, threshold)
    print("Decision: %s [%s]" % decision)
    for d in detections:
        print("  %-40s %5.1f%%" % (d["name"], d["confidence"] * 100))
    if output:
        utils.imwrite(output, frame)
        print("Saved", output)
    else:
        cv2.imshow("Traffic sign recognizer (press any key)", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video(source, recognizer, args):
    is_camera = isinstance(source, int)
    if is_camera and os.name == "nt":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)  # DirectShow is the most reliable on Windows
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit("Could not open %s. Check the camera index or the video path."
                 % ("camera %d" % source if is_camera else source))

    writer = None
    frames = 0
    last = time.time()
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break  # end of the video
        frame = resize_if_needed(frame)
        frame, detections, decision = process_frame(frame, recognizer, args.threshold)

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6) if frames else 0.0
        last = now
        cv2.putText(frame, "%.0f fps" % fps, (frame.shape[1] - 90, frame.shape[0] - 12),
                    utils.FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if args.output:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                         cap.get(cv2.CAP_PROP_FPS) or 24.0, (w, h))
            writer.write(frame)
        if not args.no_display:
            cv2.imshow("Traffic sign recognizer (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frames += 1
        if args.max_frames and frames >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
        print("Saved", args.output)
    cv2.destroyAllWindows()
    print("Processed %d frames." % frames)


def main():
    parser = argparse.ArgumentParser(description="Real-time traffic sign recognition")
    parser.add_argument("--source", default="0",
                        help="camera index (0, 1, ...), a video file, or an image file")
    parser.add_argument("--threshold", type=float, default=0.80,
                        help="minimum confidence to accept a detection (default 0.80)")
    parser.add_argument("--output", help="save the annotated video/image to this path")
    parser.add_argument("--no-display", action="store_true", help="do not open a window")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    args = parser.parse_args()

    recognizer = utils.Recognizer()
    if not recognizer.ready:
        sys.exit(recognizer.error)

    if args.source.isdigit():
        run_video(int(args.source), recognizer, args)
    elif args.source.lower().endswith(IMAGE_EXTS):
        run_image(args.source, recognizer, args.threshold, args.output)
    else:
        if not os.path.isfile(args.source):
            sys.exit("File not found: %s" % args.source)
        run_video(args.source, recognizer, args)


if __name__ == "__main__":
    main()
