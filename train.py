"""Trains the traffic sign CNN.

    python train.py                 # auto: real GTSRB if present, else synthetic
    python train.py --data gtsrb    # require real GTSRB
    python train.py --epochs 40

Outputs (in model/):
    traffic_sign_model.h5 (+ .keras copy)   the trained network
    classes.json                            which class id each output neuron means
    training_curves.png, confusion_matrix.png, metrics.json
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")  # draw plots to files, no window needed
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import utils

IMG_EXTS = (".png", ".jpg", ".jpeg", ".ppm", ".bmp")
SEED = 42


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def find_class_dirs(root):
    """Find folders named 0..42 that contain images (works for the official
    GTSRB layout '00000'..'00042' and the Kaggle layout 'Train/0'..'Train/42')."""
    found = {}
    if not os.path.isdir(root):
        return found
    for dirpath, _dirs, files in os.walk(root):
        name = os.path.basename(dirpath)
        if not (name.isdigit() and int(name) < 43):
            continue
        images = [f for f in files if f.lower().endswith(IMG_EXTS)]
        cid = int(name)
        if images and (cid not in found or len(images) > len(found[cid][1])):
            found[cid] = (dirpath, images)
    return found


def choose_dataset(mode):
    real_root = os.path.join(utils.DATA_DIR, "gtsrb")
    synth_root = os.path.join(utils.DATA_DIR, "synthetic")
    if mode in ("auto", "gtsrb"):
        dirs = find_class_dirs(real_root)
        if len(dirs) >= 30:
            return "gtsrb", dirs
        if mode == "gtsrb":
            sys.exit("GTSRB was not found in data/gtsrb. Run 'python download_data.py' first.")
    dirs = find_class_dirs(synth_root)
    if len(dirs) < 5:
        print("No dataset found, creating the synthetic one first...")
        import generate_synthetic_data
        generate_synthetic_data.generate(synth_root)
        dirs = find_class_dirs(synth_root)
    return "synthetic", dirs


def load_images(class_dirs):
    X, y = [], []
    for cid in sorted(class_dirs):
        folder, files = class_dirs[cid]
        for fname in files:
            img = utils.imread(os.path.join(folder, fname))
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(cv2.resize(rgb, (utils.IMG_SIZE, utils.IMG_SIZE), interpolation=cv2.INTER_AREA))
            y.append(cid)
        print("  class %2d: %5d images  %s" % (cid, sum(1 for v in y if v == cid), utils.class_name(cid)))
    return np.array(X, dtype=np.uint8), np.array(y, dtype=np.int64)


def load_dataset(name, class_dirs, rebuild):
    cache = os.path.join(utils.DATA_DIR, "cache_%s_%d.npz" % (name, utils.IMG_SIZE))
    if os.path.exists(cache) and not rebuild:
        data = np.load(cache)
        print("Loaded cached arrays from", cache)
        return data["X"], data["y"]
    print("Reading images (first run only, later runs use a cache)...")
    X, y = load_images(class_dirs)
    np.savez_compressed(cache, X=X, y=y)
    return X, y


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_model(num_classes):
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(utils.IMG_SIZE, utils.IMG_SIZE, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)  # input stays 0..255, scaling lives in the model
    for filters in (32, 64, 128):
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="traffic_sign_cnn")


def make_dataset(X, y_onehot, batch, training):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    # No horizontal flips: they would turn "turn left" into "turn right".
    augment = keras.Sequential([
        layers.RandomRotation(0.05, fill_mode="nearest"),            # about +-9 degrees
        layers.RandomZoom(0.15, fill_mode="nearest"),
        layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),
        layers.RandomBrightness(0.25, value_range=(0, 255)),
    ])
    ds = tf.data.Dataset.from_tensor_slices((X, y_onehot))
    if training:
        ds = ds.shuffle(len(X), seed=SEED)
    ds = ds.batch(batch)
    ds = ds.map(lambda a, b: (tf.cast(a, tf.float32), b), num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(lambda a, b: (augment(a, training=True), b), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def save_curves(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in ((axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")):
        ax.plot(history.history[key], label="train")
        ax.plot(history.history["val_" + key], label="validation")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_confusion(y_true, y_pred, class_ids, path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_ids))))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    size = max(6, min(16, len(class_ids) * 0.32))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ticks = list(range(len(class_ids)))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(c) for c in class_ids], rotation=90, fontsize=7)
    ax.set_yticklabels([str(c) for c in class_ids], fontsize=7)
    ax.set_xlabel("predicted class id")
    ax.set_ylabel("true class id")
    ax.set_title("Confusion matrix (validation set, row-normalised)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train the traffic sign CNN")
    parser.add_argument("--data", choices=["auto", "gtsrb", "synthetic"], default="auto")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--rebuild-cache", action="store_true", help="re-read all images from disk")
    args = parser.parse_args()

    np.random.seed(SEED)
    try:
        import tensorflow as tf
    except ImportError:
        sys.exit("TensorFlow is not installed. Run: pip install -r requirements.txt")
    from tensorflow import keras
    tf.random.set_seed(SEED)

    os.makedirs(utils.MODEL_DIR, exist_ok=True)
    name, class_dirs = choose_dataset(args.data)
    print("Dataset: %s (%d classes)" % (name, len(class_dirs)))
    X, y = load_dataset(name, class_dirs, args.rebuild_cache)

    class_ids = sorted(set(int(c) for c in y))
    index_of = {cid: i for i, cid in enumerate(class_ids)}
    y_idx = np.array([index_of[int(c)] for c in y], dtype=np.int64)
    num_classes = len(class_ids)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_idx, test_size=0.2, random_state=SEED, stratify=y_idx)
    print("Train: %d images, validation: %d images" % (len(X_train), len(X_val)))

    train_ds = make_dataset(X_train, keras.utils.to_categorical(y_train, num_classes), args.batch, True)
    val_ds = make_dataset(X_val, keras.utils.to_categorical(y_val, num_classes), args.batch, False)

    model = build_model(num_classes)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=6,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(utils.BEST_PATH, monitor="val_accuracy", mode="max",
                                        save_best_only=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                                          min_lr=1e-5, verbose=1),
    ]
    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        callbacks=callbacks, verbose=2)
    print("Training took %.1f minutes" % ((time.time() - start) / 60.0))

    # EarlyStopping restored the best weights, so this is the best model.
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print("\nValidation accuracy: %.2f%%  (loss %.4f)" % (val_acc * 100, val_loss))

    try:
        model.save(utils.H5_PATH)
    except Exception as exc:  # very new Keras versions may refuse the legacy .h5 format
        print("Could not write the .h5 file (%s). The .keras file is used instead." % exc)
    model.save(utils.KERAS_PATH)

    with open(utils.CLASSES_PATH, "w", encoding="utf-8") as fh:
        json.dump({"class_ids": class_ids, "img_size": utils.IMG_SIZE, "dataset": name}, fh, indent=2)

    probs = model.predict(val_ds, verbose=0)
    y_pred = probs.argmax(axis=1)
    names = ["%d %s" % (c, utils.class_name(c)[:30]) for c in class_ids]
    print(classification_report(y_val, y_pred, target_names=names, digits=3, zero_division=0))

    save_curves(history, os.path.join(utils.MODEL_DIR, "training_curves.png"))
    save_confusion(y_val, y_pred, class_ids, os.path.join(utils.MODEL_DIR, "confusion_matrix.png"))
    with open(os.path.join(utils.MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({"dataset": name, "classes": num_classes, "val_accuracy": float(val_acc),
                   "val_loss": float(val_loss), "epochs_run": len(history.history["loss"])}, fh, indent=2)

    print("Saved model and plots in", utils.MODEL_DIR)
    if val_acc < 0.95:
        print("Note: accuracy is below 95%. Try --epochs 40 or use the real GTSRB dataset.")


if __name__ == "__main__":
    main()
