"""Train a small U-Net from paired dental images and binary masks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

SIZE = (256, 256)


def pairs(split_dir: Path):
    image_dir, mask_dir = split_dir / "image", split_dir / "mask"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected image/ and mask/ folders under {split_dir}")
    masks = {path.stem: path for path in mask_dir.iterdir() if path.is_file()}
    result = [(path, masks[path.stem]) for path in image_dir.iterdir()
              if path.is_file() and path.stem in masks]
    if not result:
        raise ValueError(f"No image/mask pairs with matching filenames found in {split_dir}")
    return result


def dataset(split_dir: Path, batch_size: int, training: bool):
    matched = pairs(split_dir)
    images = [str(image) for image, _ in matched]
    masks = [str(mask) for _, mask in matched]
    ds = tf.data.Dataset.from_tensor_slices((images, masks))

    def load(image_path, mask_path):
        image = tf.io.decode_image(tf.io.read_file(image_path), channels=3, expand_animations=False)
        mask = tf.io.decode_image(tf.io.read_file(mask_path), channels=1, expand_animations=False)
        image = tf.image.resize(image, SIZE) / 255.0
        mask = tf.image.resize(mask, SIZE, method="nearest")
        mask = tf.cast(mask > 127, tf.float32)
        return image, mask

    if training:
        ds = ds.shuffle(len(matched), seed=42, reshuffle_each_iteration=True)
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


@tf.keras.utils.register_keras_serializable(package="TeethSegmentation")
def dice_loss(y_true, y_pred):
    smooth = 1.0
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    total = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    return 1.0 - tf.reduce_mean((2.0 * intersection + smooth) / (total + smooth))


@tf.keras.utils.register_keras_serializable(package="TeethSegmentation")
def combined_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def unet():
    inputs = tf.keras.Input((*SIZE, 3))
    x = inputs
    skips = []
    for filters in (32, 64, 128, 256):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        skips.append(x)
        x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu")(x)
    for filters, skip in zip((256, 128, 64, 32), reversed(skips)):
        x = tf.keras.layers.UpSampling2D(interpolation="bilinear")(x)
        x = tf.keras.layers.Concatenate()([x, skip])
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return tf.keras.Model(inputs, tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(x))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Folder containing train/ and val/ image/ and mask/ pairs")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    train = dataset(args.data / "train", args.batch_size, True)
    val = dataset(args.data / "val", args.batch_size, False)
    model = unet()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=combined_loss)
    output = Path("models")
    output.mkdir(parents=True, exist_ok=True)
    model.fit(train, validation_data=val, epochs=args.epochs, callbacks=[
        tf.keras.callbacks.ModelCheckpoint(output / "segmentation.keras", monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    ])
    metrics = {"train_pairs": len(pairs(args.data / "train")), "validation_pairs": len(pairs(args.data / "val"))}
    test_dir = args.data / "test"
    if test_dir.exists():
        metrics["test_pairs"] = len(pairs(test_dir))
        metrics["test_loss"] = float(model.evaluate(dataset(test_dir, args.batch_size, False), verbose=0))
    (output / "segmentation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
