"""Prepare the Teeth_Dataset ZIP, retaining only valid split/class images."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

SPLITS = ("Training", "Validation", "Testing")
CLASSES = ("CaS", "CoS", "Gum", "MC", "OC", "OLP", "OT")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Downloaded Teeth DataSet.zip")
    parser.add_argument("--output", type=Path, default=Path("data/teeth_clean"))
    args = parser.parse_args()
    if not args.archive.is_file():
        parser.error(f"Archive not found: {args.archive}")

    selected = []
    original_counts: Counter[tuple[str, str]] = Counter()
    skipped = 0
    with zipfile.ZipFile(args.archive) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"Corrupt archive member: {bad}")
        for member in archive.infolist():
            if member.is_dir():
                continue
            path = PurePosixPath(member.filename)
            parts = path.parts
            if len(parts) != 4 or parts[1] not in SPLITS or parts[2] not in CLASSES or path.suffix.lower() not in EXTENSIONS:
                skipped += 1
                continue
            # Never extract paths from the archive; stream each selected image to its own destination.
            original_split, class_name = parts[1], parts[2]
            original_counts[(original_split, class_name)] += 1
            match = re.match(r"^[^_]+_(\d+)", path.stem)
            if not match:
                raise RuntimeError(f"Cannot derive a patient/source group from filename: {path.name}")
            selected.append((member, class_name, match.group(1), original_split))

    groups = np.asarray([item[2] for item in selected])
    labels = np.asarray([CLASSES.index(item[1]) for item in selected])
    indices = np.arange(len(selected))
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    trainval_idx, test_idx = next(outer.split(indices, labels, groups))
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=123)
    train_idx, val_idx = next(inner.split(trainval_idx, labels[trainval_idx], groups[trainval_idx]))
    # `inner` returns positions inside trainval_idx.
    assignments = {int(trainval_idx[i]): "Training" for i in train_idx}
    assignments.update({int(trainval_idx[i]): "Validation" for i in val_idx})
    assignments.update({int(i): "Testing" for i in test_idx})
    counts: Counter[tuple[str, str]] = Counter()
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    with zipfile.ZipFile(args.archive) as archive:
        for index, (member, class_name, group, _) in enumerate(selected):
            split = assignments[index]
            dest = args.output / split / class_name / PurePosixPath(member.filename).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, dest.open("wb") as target:
                shutil.copyfileobj(source, target)
            counts[(split, class_name)] += 1
            groups_by_split[split].add(group)
    overlap = (groups_by_split["Training"] & groups_by_split["Validation"]
               | groups_by_split["Training"] & groups_by_split["Testing"]
               | groups_by_split["Validation"] & groups_by_split["Testing"])
    if overlap:
        raise RuntimeError(f"Patient/source groups leaked across new splits: {sorted(overlap)}")
    expected = {(split, cls) for split in SPLITS for cls in CLASSES}
    missing = sorted(expected - counts.keys())
    if missing:
        raise RuntimeError(f"Empty split/class folders after group split: {missing}")
    report = {
        "classes": list(CLASSES),
        "counts": {f"{split}/{cls}": counts[(split, cls)] for split in SPLITS for cls in CLASSES},
        "ignored_non_class_files": skipped,
        "original_split_counts": {f"{split}/{cls}": original_counts[(split, cls)] for split in SPLITS for cls in CLASSES},
        "source_groups_per_split": {split: len(groups_by_split[split]) for split in SPLITS},
        "source_group_overlap_after_resplit": len(overlap),
        "split_method": "two-stage stratified group split by numeric filename ID (approx 64/16/20)",
        "note": "The source archive reused numeric filename IDs across its supplied splits; these files are regrouped to reduce leakage.",
    }
    (args.output / "data_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
