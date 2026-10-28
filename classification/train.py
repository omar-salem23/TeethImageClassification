"""Train and evaluate an EfficientNetB0 dental image classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

CLASSES = ["CaS", "CoS", "Gum", "MC", "OC", "OLP", "OT"]
SIZE = (224, 224)


def load_split(root: Path, split: str, batch_size: int, shuffle: bool):
    return tf.keras.utils.image_dataset_from_directory(
        root / split, class_names=CLASSES, image_size=SIZE, batch_size=batch_size,
        shuffle=shuffle, seed=42,
    ).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/teeth_clean"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("models"))
    args = parser.parse_args()
    train = load_split(args.data, "Training", args.batch_size, True)
    val = load_split(args.data, "Validation", args.batch_size, False)
    test = load_split(args.data, "Testing", args.batch_size, False)

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(*SIZE, 3)
    )
    backbone.trainable = False
    inputs = tf.keras.Input(shape=(*SIZE, 3))
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.05)(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    args.output.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.output / "classification.keras",
                                           monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                         restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(args.output / "classification_training.csv"),
    ]
    model.fit(train, validation_data=val, epochs=args.epochs, callbacks=callbacks)
    model = tf.keras.models.load_model(args.output / "classification.keras")
    y_true = np.concatenate([labels.numpy() for _, labels in test])
    probs = model.predict(test, verbose=0)
    y_pred = probs.argmax(axis=1)
    report = classification_report(y_true, y_pred, target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    (args.output / "classification_metrics.json").write_text(
        json.dumps({"classes": CLASSES, "classification_report": report,
                    "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()}, indent=2),
        encoding="utf-8",
    )
    (args.output / "class_names.json").write_text(json.dumps(CLASSES), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
