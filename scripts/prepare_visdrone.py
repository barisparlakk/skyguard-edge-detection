#!/usr/bin/env python3
"""Download VisDrone2019-DET train/val and convert annotations to YOLO format.

The output layout is:

    data/visdrone/
    ├── images/{train,val}/
    ├── labels/{train,val}/
    └── raw/VisDrone2019-DET-{train,val}/annotations/

Raw annotations are retained so conversion remains auditable and repeatable.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image


# These are the dataset mirrors used by Ultralytics' VisDrone dataset config.
ASSET_ROOT = "https://github.com/ultralytics/assets/releases/download/v0.0.0"
ARCHIVES = {
    "train": f"{ASSET_ROOT}/VisDrone2019-DET-train.zip",
    "val": f"{ASSET_ROOT}/VisDrone2019-DET-val.zip",
}

# VisDrone categories 1-10 become YOLO class IDs 0-9 in this order.
CLASS_NAMES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)


def download_file(url: str, destination: Path) -> None:
    """Download once to a temporary file, then atomically rename it."""
    if destination.exists():
        print(f"Using existing archive: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}")
    try:
        urllib.request.urlretrieve(url, temporary)  # noqa: S310 - URL is fixed above.
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract a ZIP while rejecting paths that escape the destination."""
    if destination.exists():
        print(f"Using existing extracted data: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved_root = destination.parent.resolve()
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            member_path = (destination.parent / member.filename).resolve()
            if not member_path.is_relative_to(resolved_root):
                raise ValueError(f"Unsafe path in {archive}: {member.filename}")
        print(f"Extracting {archive}")
        zip_file.extractall(destination.parent)

    if not destination.exists():
        raise FileNotFoundError(
            f"Archive did not create expected directory: {destination}"
        )


def install_images(source: Path, destination: Path) -> None:
    """Move images into YOLO's images/{split} layout without duplicating data."""
    destination.mkdir(parents=True, exist_ok=True)
    source_images = source / "images"

    if source_images.exists():
        for image_path in source_images.glob("*.jpg"):
            target = destination / image_path.name
            if not target.exists():
                shutil.move(str(image_path), target)

    if not any(destination.glob("*.jpg")):
        raise FileNotFoundError(f"No JPG images found in {destination}")


def convert_split(source: Path, images: Path, labels: Path) -> tuple[int, int]:
    """Convert one VisDrone split; return image and retained-box counts."""
    annotation_paths = sorted((source / "annotations").glob("*.txt"))
    if not annotation_paths:
        raise FileNotFoundError(f"No annotations found in {source / 'annotations'}")

    labels.mkdir(parents=True, exist_ok=True)
    retained_boxes = 0

    for annotation_path in annotation_paths:
        image_path = images / f"{annotation_path.stem}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for {annotation_path}: {image_path}")

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        yolo_rows: list[str] = []
        for line_number, line in enumerate(
            annotation_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue

            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 8:
                raise ValueError(
                    f"{annotation_path}:{line_number}: expected 8 fields, got {len(fields)}"
                )

            left, top, width, height = map(float, fields[:4])
            score = int(fields[4])
            category = int(fields[5])

            # Ignore regions, "others", and boxes excluded by VisDrone evaluation.
            if score == 0 or category not in range(1, len(CLASS_NAMES) + 1):
                continue
            if width <= 0 or height <= 0:
                continue

            x_center = (left + width / 2.0) / image_width
            y_center = (top + height / 2.0) / image_height
            width_normalized = width / image_width
            height_normalized = height / image_height
            class_id = category - 1

            values = (x_center, y_center, width_normalized, height_normalized)
            if not all(0.0 <= value <= 1.0 for value in values):
                raise ValueError(
                    f"{annotation_path}:{line_number}: box lies outside image bounds"
                )

            yolo_rows.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} "
                f"{width_normalized:.6f} {height_normalized:.6f}\n"
            )
            retained_boxes += 1

        (labels / annotation_path.name).write_text(
            "".join(yolo_rows), encoding="utf-8"
        )

    return len(annotation_paths), retained_boxes


def prepare_split(root: Path, split: str) -> None:
    """Download, extract, arrange, and convert one dataset split."""
    archive = root / "downloads" / f"VisDrone2019-DET-{split}.zip"
    source = root / "raw" / f"VisDrone2019-DET-{split}"
    images = root / "images" / split
    labels = root / "labels" / split

    download_file(ARCHIVES[split], archive)
    safe_extract(archive, source)
    install_images(source, images)
    image_count, box_count = convert_split(source, images, labels)
    print(f"Prepared {split}: {image_count} images, {box_count} retained boxes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/visdrone"),
        help="Dataset output root (default: data/visdrone)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split in ("train", "val"):
        prepare_split(args.output.resolve(), split)


if __name__ == "__main__":
    main()
