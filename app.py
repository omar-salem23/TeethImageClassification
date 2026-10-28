"""Streamlit demo for dental classification, detection, or segmentation models."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
CLASS_NAMES_PATH = MODELS / "class_names.json"

st.set_page_config(page_title="Dental Image AI", page_icon="🦷", layout="wide")
st.title("Dental Image AI")
st.caption("Experimental research demo. This tool is not a medical diagnosis.")

task = st.radio(
    "Choose a model task",
    ["Classification", "Object detection (bounding boxes)", "Segmentation"],
    horizontal=True,
)
uploaded = st.file_uploader("Upload a dental image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    left, right = st.columns(2)
    with left:
        st.image(image, caption="Uploaded image", use_container_width=True)
    with right:
        if task == "Classification":
            model_path = MODELS / "classification.keras"
            if not model_path.is_file() or not CLASS_NAMES_PATH.is_file():
                st.warning("Classification weights are not trained yet. Run the classification training script first.")
            else:
                import tensorflow as tf

                model = tf.keras.models.load_model(model_path, compile=False)
                names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
                height, width = model.input_shape[1:3]
                sample = image.resize((width, height))
                x = np.asarray(sample, dtype=np.float32)[None, ...]
                scores = model.predict(x, verbose=0)[0]
                order = np.argsort(scores)[::-1]
                st.subheader(names[int(order[0])])
                st.metric("Model confidence", f"{scores[order[0]]:.1%}")
                st.dataframe(
                    [{"Class": names[int(i)], "Score": float(scores[i])} for i in order],
                    use_container_width=True,
                    hide_index=True,
                )
        elif task.startswith("Object detection"):
            model_path = MODELS / "detection.pt"
            if not model_path.is_file():
                st.warning("Object detection weights are not available. Train on an annotated bounding-box dataset first.")
            else:
                from ultralytics import YOLO

                model = YOLO(str(model_path))
                result = model.predict(image, conf=0.25, verbose=False)[0]
                annotated = result.plot()[:, :, ::-1]
                st.image(annotated, caption="Predicted boxes", use_container_width=True)
                if result.boxes is None or len(result.boxes) == 0:
                    st.info("No objects were detected above the confidence threshold.")
                else:
                    rows = []
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        rows.append({
                            "Class": model.names[class_id],
                            "Confidence": float(box.conf.item()),
                            "Box (x1, y1, x2, y2)": [round(v, 1) for v in box.xyxy[0].tolist()],
                        })
                    st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            model_path = MODELS / "segmentation.keras"
            if not model_path.is_file():
                st.warning("Segmentation weights are not available. Train on paired dental images and masks first.")
            else:
                import tensorflow as tf

                model = tf.keras.models.load_model(model_path, compile=False)
                height, width = model.input_shape[1:3]
                sample = image.resize((width, height))
                x = np.asarray(sample, dtype=np.float32)[None, ...] / 255.0
                mask = model.predict(x, verbose=0)[0]
                if mask.ndim == 3:
                    mask = mask[..., 0]
                mask = Image.fromarray((mask > 0.5).astype(np.uint8) * 255).resize(image.size)
                overlay = image.copy().convert("RGBA")
                color = Image.new("RGBA", image.size, (255, 0, 0, 0))
                color.putalpha(mask.point(lambda p: 110 if p else 0))
                st.image(Image.alpha_composite(overlay, color).convert("RGB"),
                         caption="Predicted segmentation mask", use_container_width=True)
