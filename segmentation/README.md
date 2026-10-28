# Dental image segmentation

Segmentation needs a mask for every training image. The linked segmentation
notebook uses a U-Net with paired files under `train/image` and `train/mask`, but
its data path is **Eyes data**, not dental images. Those eye masks cannot train a
dental segmentation model. The linked notebook also evaluates on a validation
generator rather than an independent held-out test set.

Provide corresponding dental image/mask pairs before training. The Streamlit app
loads a binary Keras mask model from `models/segmentation.keras`.
