# Dental object detection

The supplied `Teeth_Dataset.zip` contains image-level class folders only. It has no
bounding-box labels, so it cannot train an object detector as-is. Obtain a dental
dataset with YOLO-format boxes (`images/{train,val,test}` and `labels/{train,val,test}`)
before training. Keep each patient's images in one split to avoid leakage.

When labels are ready, train with:

```bash
yolo detect train model=yolo11n.pt data=path/to/data.yaml imgsz=640 epochs=100
```

Copy the selected `best.pt` to `models/detection.pt` for the Streamlit app.
