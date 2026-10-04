"""Classify one traffic sign image from the command line.

    python predict.py samples\\sign_14_stop.png
    python predict.py my_photo.jpg --top 5

The image should be a close-up of a single sign. For a whole street scene use
detect_video.py --source scene.jpg (or the "Road scene" mode of the web app).
"""
import argparse
import os
import sys

import utils


def main():
    parser = argparse.ArgumentParser(description="Recognize a traffic sign in one image")
    parser.add_argument("image", help="path to a PNG/JPG/BMP image of a single sign")
    parser.add_argument("--top", type=int, default=3, help="how many guesses to show")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        sys.exit("File not found: %s" % args.image)
    img = utils.imread(args.image)
    if img is None:
        sys.exit("Could not read %s as an image. Use PNG, JPG or BMP." % args.image)

    recognizer = utils.Recognizer()
    if not recognizer.ready:
        sys.exit(recognizer.error)

    result = recognizer.classify_batch([img], k=args.top)[0]
    print()
    print("Sign      : %s (class %d)" % (result["name"], result["class_id"]))
    print("Confidence: %.1f%%" % (result["confidence"] * 100))
    print("Decision  : %s [%s]" % (result["action"], result["level"]))
    print("\nTop guesses:")
    for item in result["top"]:
        print("  %5.1f%%  %s" % (item["confidence"] * 100, item["name"]))
    if result["confidence"] < 0.5:
        print("\nThe model is unsure. Try a sharper, closer photo of the sign.")


if __name__ == "__main__":
    main()
