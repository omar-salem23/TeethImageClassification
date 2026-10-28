"""Train YOLO on a separately annotated dental detection dataset."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_yaml", type=Path, help="YOLO data.yaml with train/val images and class names")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="cpu, 0, or auto")
    parser.add_argument("--output", type=Path, default=Path("runs/detect/teeth"))
    args = parser.parse_args()
    if not args.data_yaml.is_file():
        parser.error(f"YOLO data file not found: {args.data_yaml}")
    model = YOLO(args.weights)
    model.train(data=str(args.data_yaml), epochs=args.epochs, imgsz=args.imgsz,
                device=None if args.device == "auto" else args.device,
                project=str(args.output.parent), name=args.output.name)
    best = args.output / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training finished without expected checkpoint: {best}")
    destination = Path("models/detection.pt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, destination)
    print(f"Selected weights saved to {destination}")


if __name__ == "__main__":
    main()
