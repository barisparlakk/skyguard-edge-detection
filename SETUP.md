# Skyguard setup

This project uses NVIDIA's NGC PyTorch container so PyTorch, CUDA, cuDNN, and
TensorRT come from a tested GPU software stack. Install a compatible NVIDIA
driver and Docker with NVIDIA Container Toolkit on the GCP L4 instance first.

## 1. Verify the L4 on the host

```bash
nvidia-smi
```

The output should list one `NVIDIA L4` GPU. If it does not, fix the GCP GPU,
driver, or NVIDIA Container Toolkit setup before starting the container.

## 2. Launch the NGC PyTorch container

Run this command from the repository root:

```bash
docker run --gpus all --ipc=host --rm -it \
  -v "$(pwd):/workspace/skyguard" \
  -w /workspace/skyguard \
  nvcr.io/nvidia/pytorch:24.05-py3
```

- `--gpus all` exposes the L4 to the container.
- `--ipc=host` gives PyTorch data-loader workers enough shared memory.
- `--rm` removes the stopped container; project files remain in the bind mount.
- `-v` mounts this repository so outputs persist on the GCP instance.
- `-w` starts the shell in the mounted repository.

Inside the container, verify GPU access again and inspect the Python version:

```bash
nvidia-smi
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"
```

The final command should print the container's PyTorch version and `NVIDIA L4`.
The `24.05-py3` image controls the exact Python/PyTorch/CUDA versions; check
`python --version` rather than assuming the tag provides Python 3.11. If Python
3.11 is a strict requirement, select or build an NGC image that explicitly
provides it while preserving CUDA and TensorRT compatibility.

Install the project-level Python packages:

```bash
# The regular OpenCV wheel used by Ultralytics links against these runtime
# libraries, which are absent from the headless NGC base image.
apt-get update
apt-get install -y libgl1 libglib2.0-0

# NGC 24.05 includes an `opencv` package that owns the same `cv2` directory as
# Ultralytics' `opencv-python` dependency. Remove it to avoid mixed binaries.
python -m pip uninstall -y opencv
python -m pip install -r requirements.txt
```

Package roles:

- `ultralytics`: trains and validates YOLOv8m and exports the trained model.
- `onnx`: represents and validates the portable exported model graph.
- `onnxruntime-gpu`: runs the ONNX graph through CUDA for the ONNX benchmark.
- `numpy`: handles input arrays and computes benchmark statistics.
- `Pillow`: reads image dimensions while converting VisDrone boxes to YOLO labels.

## Why ONNX is portable but TensorRT engines are not

ONNX stores a framework-neutral computation graph, weights, data types, and
operator definitions (this project will export with opset 17). A compatible
runtime can load that graph on different machines and choose its own execution
provider. Portability still depends on runtime support for every exported ONNX
operator, so the model must be validated after export.

A TensorRT engine is the compiled result of profiling and selecting kernels for
a specific target environment. Its tactics can depend on GPU architecture,
TensorRT version, CUDA stack, precision support, and configured input shapes.
Build and benchmark the FP16 and INT8 engines on the target L4 instance instead
of building them on a local machine. Rebuild engines after meaningful changes to
the GPU or TensorRT/CUDA environment. INT8 additionally requires representative
calibration data unless the model already carries usable quantization scales.

## Common setup mistakes

- Starting Docker without `--gpus all`, which hides the L4 from the container.
- Running from the wrong directory, which mounts the wrong folder at `/workspace`.
- Installing a separate CUDA toolkit with pip or apt and breaking the NGC stack.
- Treating a copied TensorRT engine as portable across GPU/software environments.
- Committing datasets, checkpoints, ONNX files, or TensorRT engines to Git.

Dataset download and model training are intentionally deferred to later phases.
