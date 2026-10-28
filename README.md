# Dental image AI

This repository contains three separate computer-vision tracks and a Streamlit
demo that lets you choose the task. The current `Teeth_Dataset.zip` supports only
seven-way image classification. It has no bounding boxes or segmentation masks.
See [`docs/resource_review.md`](docs/resource_review.md) for the review of the
existing notebook, dataset integrity, and the linked tutorials.

## What's in the repository

- `classification/`: EfficientNetB0 training and held-out evaluation for the seven
  class folders: CaS, CoS, Gum, MC, OC, OLP, and OT.
- `detection/`: setup notes for object detection. It needs images with bounding-box
  annotations before a detector can be trained.
- `segmentation/`: setup notes for segmentation. It needs paired dental images and
  masks. The linked U-Net notebook points to an **eye** dataset, so those weights
  would not be suitable for dental images.
- `app.py`: upload an image and select classification, bounding-box detection, or
  segmentation. It shows a clear message when weights for a task are unavailable.

## Prepare and train classification

Install dependencies and prepare the dataset archive. The preparation script
ignores output folders accidentally included in the ZIP and makes a new split by
the numeric image/source ID because the original split reuses IDs across train,
validation, and test.

```bash
python -m pip install -r requirements.txt
python tools/prepare_teeth_data.py "/path/to/Teeth DataSet.zip"
python classification/train.py --data data/teeth_clean
streamlit run app.py
```

The best classifier is saved as `models/classification.keras`. The metrics and
confusion matrix are saved next to it. The classifier is an experimental
educational model, not a diagnostic tool.

## Detection and segmentation

The app already supports inference for weights placed at
`models/detection.pt` (Ultralytics) and `models/segmentation.keras` (binary Keras
mask model). Those models cannot be trained from the supplied teeth archive.
Provide dental box labels and paired dental masks to enable those tracks. The
Drive videos are tutorials, not labeled training data.
