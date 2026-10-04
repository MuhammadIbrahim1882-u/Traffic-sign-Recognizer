"""Downloads the GTSRB training images (about 260 MB) into data/gtsrb.

If the download fails (no internet, blocked link, moved file) the script
falls back to a synthetic dataset so the project still trains and runs.

Usage:
    python download_data.py              # try GTSRB, fall back to synthetic
    python download_data.py --synthetic  # skip the download, use synthetic data

Manual option: download GTSRB from Kaggle ("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
and unzip it so that folders named 0..42 sit somewhere under data/gtsrb/
(for example data/gtsrb/Train/0, data/gtsrb/Train/1, ...).
"""
import argparse
import os
import shutil
import sys
import urllib.request
import zipfile

import utils

GTSRB_URLS = [
    "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip",
]
IMG_EXTS = (".png", ".jpg", ".jpeg", ".ppm", ".bmp")


def count_class_folders(root):
    """How many class folders (named 0..42) with images exist under root."""
    found = set()
    for dirpath, _dirs, files in os.walk(root):
        name = os.path.basename(dirpath)
        if name.isdigit() and int(name) < 43 and any(f.lower().endswith(IMG_EXTS) for f in files):
            found.add(int(name))
    return len(found)


def download(url, target):
    print("Downloading", url)
    with urllib.request.urlopen(url, timeout=30) as resp, open(target, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write("\r  %.0f%% (%d / %d MB)" % (100.0 * done / total, done >> 20, total >> 20))
                sys.stdout.flush()
    print()


def fetch_gtsrb(dest):
    os.makedirs(dest, exist_ok=True)
    zip_path = os.path.join(dest, "gtsrb_train.zip")
    for url in GTSRB_URLS:
        try:
            download(url, zip_path)
            print("Extracting...")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest)
            os.remove(zip_path)
            n = count_class_folders(dest)
            if n >= 30:
                print("GTSRB ready: %d class folders in %s" % (n, dest))
                return True
            print("The archive did not contain the expected class folders.")
        except Exception as exc:  # network errors, bad zip, disk problems
            print("Download failed: %s" % exc)
        if os.path.exists(zip_path):
            os.remove(zip_path)
    return False


def main():
    parser = argparse.ArgumentParser(description="Get the traffic sign dataset")
    parser.add_argument("--synthetic", action="store_true", help="skip GTSRB, create synthetic data")
    parser.add_argument("--per-class", type=int, default=300, help="images per class for synthetic data")
    args = parser.parse_args()

    gtsrb_dir = os.path.join(utils.DATA_DIR, "gtsrb")
    synthetic_dir = os.path.join(utils.DATA_DIR, "synthetic")
    os.makedirs(utils.DATA_DIR, exist_ok=True)

    if not args.synthetic:
        if os.path.isdir(gtsrb_dir) and count_class_folders(gtsrb_dir) >= 30:
            print("GTSRB is already in %s, nothing to download." % gtsrb_dir)
            return
        if fetch_gtsrb(gtsrb_dir):
            return
        shutil.rmtree(gtsrb_dir, ignore_errors=True)
        print("\nCould not get GTSRB automatically. Using the synthetic dataset instead.")
        print("(You can still add GTSRB by hand later; see the note at the top of this file.)\n")

    import generate_synthetic_data
    generate_synthetic_data.generate(synthetic_dir, args.per_class)


if __name__ == "__main__":
    main()
