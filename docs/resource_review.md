# Review of the supplied project materials

## Teeth classification archive

The Google Drive ZIP is readable and passes its ZIP integrity check. It contains
5,624 JPG files. Of those, 5,143 are in the seven named dental classes across the
provided splits; 481 files are stray `output`/`outputs` exports and are ignored by
the preparation script. The seven classes are `CaS`, `CoS`, `Gum`, `MC`, `OC`,
`OLP`, and `OT`.

The original notebook builds a small CNN from scratch, trains for a fixed 15
epochs, reports accuracy only, and never saves a model checkpoint. It also has an
augmentation loop over Training, Validation, and Testing. Augmenting the held-out
splits would make evaluation unreliable if those augmented images are used.

The archive's filename IDs recur across its supplied splits. The audit found 172
numeric IDs present in more than one split, which can leak related/source images
into evaluation. The new preparation script instead groups by the numeric ID and
creates new stratified train/validation/test groups with zero ID overlap. This is
a safer estimate, though filename grouping cannot prove true patient identity.

The first CPU-only run with a randomly initialized CNN reached 36.9% accuracy on
the grouped held-out test set. A pretrained MobileNet transfer-learning model
then reached 57.0% accuracy and 0.545 macro F1 on the same 1,040-image grouped
test set. Its per-class recall ranges from 0.20 to 0.99, so performance is uneven
and it remains an experiment rather than a reliable diagnostic classifier. The
model uses best validation-loss checkpointing and early stopping.

## Videos and notebook links

- The Object Detection folder is accessible and contains two lecture videos. They
  cover localization, region proposals, classification, confidence, IoU, and
  non-max suppression. They are instructional videos, not a labeled dental dataset.
- The Segmentation folder is accessible and contains two U-Net lecture videos, a
  code video, and `Binary Seg_code.ipynb`.
- The segmentation notebook expects `Eyes data/train/image` and `mask`. It is an
  eye segmentation exercise, not a dental segmentation dataset. It resizes image
  and mask separately and uses pixel accuracy; it does not report Dice/IoU or
  evaluate on an independent test set.
- The first Drive folder link could not be opened from this environment, so its
  contents have not been reviewed.

## What is needed for all three Streamlit choices

The app and training entry points are laid out for all three tasks, but only image
classification data was supplied for teeth. Detection requires per-object box
annotations. Segmentation requires paired dental images and masks. Neither can be
trained from class-folder labels, and the eye masks from the tutorial cannot be
used as dental labels.
