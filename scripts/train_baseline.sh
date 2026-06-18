#!/usr/bin/env bash
# Train the reproducible YOLOv8m baseline on one NVIDIA L4.

set -euo pipefail

# Run from the repository root so visdrone.yaml resolves its relative paths.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Fail early with a useful message instead of waiting for Ultralytics to scan.
for split in train val; do
  if [[ ! -d "data/visdrone/images/${split}" ]]; then
    echo "Missing data/visdrone/images/${split}. Run scripts/prepare_visdrone.py first." >&2
    exit 1
  fi
done

# Keep the baseline destination stable and avoid silently creating/overwriting a run.
run_dir="runs/detect/baseline_yolov8m"
if [[ -e "${run_dir}" ]]; then
  echo "${run_dir} already exists; move it or resume it explicitly before retrying." >&2
  exit 1
fi

yolo detect train \
  model=yolov8m.pt \
  data=visdrone.yaml \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=8 \
  project=runs/detect \
  name=baseline_yolov8m
