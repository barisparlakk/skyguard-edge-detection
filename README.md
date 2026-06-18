# Skyguard — Aerial Object Detection on the Edge 

> Real-time aerial/drone object detection (YOLOv8m) optimized for the edge — **ONNX → TensorRT (FP16/INT8)** on an NVIDIA L4, with documented latency & throughput benchmarks.
> 

<img width="84" height="20" alt="image" src="https://github.com/user-attachments/assets/3a7ced36-b9cd-4617-89df-e563f6c9acd0" />

<img width="80" height="20" alt="image" src="https://github.com/user-attachments/assets/4970283f-f059-40fa-87e3-9042e43beba3" />

<img width="130" height="20" alt="image" src="https://github.com/user-attachments/assets/5045e716-aa58-470e-bab6-d7623a88ccab" />

<img width="78" height="20" alt="image" src="https://github.com/user-attachments/assets/6a1a5c34-4842-4333-9d35-72c7862c6431" />


---

## Overview

Skyguard is a compact, end-to-end perception project that takes a YOLOv8m detector from training to a **production-fast edge engine**. The focus is the deployment pipeline: export the trained model to ONNX, optimize it with TensorRT on an NVIDIA L4, and **measure the speedup** with a clean PyTorch → ONNX Runtime → TensorRT comparison.

**Use case:** drone's-eye-view detection of vehicles and pedestrians (Baykar-style aerial perception).

## Features

- 🎯 YOLOv8m detector trained on the **VisDrone2019** aerial dataset
- ⚡ **ONNX export** (opset 17, dynamic batch)
- 🚀 **TensorRT** FP16 and INT8 engine builds
- 📊 Reproducible **latency / throughput benchmarks** (with correct warmup + CUDA sync)
- 📝 Documented methodology and results table

## Repo structure

```
skyguard-edge-detection/
├─ data/            # VisDrone (converted to YOLO format) + data.yaml
├─ runs/            # training outputs / checkpoints
├─ export/          # onnx + tensorrt engines
├─ bench/           # benchmark scripts + raw results
├─ report/          # write-up, plots, methodology
└─ README.md
```

## Environment

Use the NVIDIA NGC PyTorch container so CUDA + TensorRT are pre-matched:

```bash
docker run --gpus all -it --rm -v $(pwd):/workspace nvcr.io/nvidia/pytorch:24.05-py3
pip install ultralytics onnx onnxruntime-gpu
```

## Dataset

Download **VisDrone2019** (detection split) and convert annotations to YOLO format, then point `data.yaml` at the train/val image folders and class names.

## Training

```bash
yolo detect train model=yolov8m.pt data=visdrone.yaml epochs=50 imgsz=640
```

Record `mAP@50` and `mAP@50-95` from validation; the best checkpoint lands in `runs/detect/train/weights/best.pt`.

## Export to ONNX

```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx opset=17 dynamic=True
```

```python
import onnx
onnx.checker.check_model(onnx.load("best.onnx"))  # validate
```

## Optimize with TensorRT

```bash
# FP16 — the sweet spot on L4
trtexec --onnx=best.onnx --saveEngine=best_fp16.engine --fp16

# INT8 — fastest, needs a calibration set
trtexec --onnx=best.onnx --saveEngine=best_int8.engine --int8
```

## Benchmark

```bash
trtexec --loadEngine=best_fp16.engine --iterations=1000 --avgRuns=100
```

PyTorch baseline timing (warmup + sync are essential):

```python
import torch, time, numpy as np
from ultralytics import YOLO
m = YOLO("best.pt")
x = torch.randn(1, 3, 640, 640).cuda()
for _ in range(20):
    _ = m.model(x)            # warmup
torch.cuda.synchronize()      # GPU calls are async
t = []
for _ in range(1000):
    s = time.perf_counter()
    _ = m.model(x)
    torch.cuda.synchronize()
    t.append(time.perf_counter() - s)
t = np.array(t) * 1000
print(f"mean {t.mean():.2f} ms | p99 {np.percentile(t,99):.2f} ms | {1000/t.mean():.1f} inf/s")
```

## Results

| Config | Precision | Latency mean (ms) | Latency p99 (ms) | Throughput (inf/s) | mAP@50 |
| --- | --- | --- | --- | --- | --- |
| PyTorch (L4, eager) | FP32 | – | – | – | – |
| ONNX Runtime (GPU) | FP32 | – | – | – | – |
| TensorRT | FP16 | – | – | – | – |
| TensorRT | INT8 | – | – | – | – |

*All rows use 640×640 input and the same batch size. GPU: NVIDIA L4. Report driver + TensorRT versions for reproducibility.*

## License

MIT
