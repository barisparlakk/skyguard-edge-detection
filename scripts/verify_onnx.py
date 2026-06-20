#!/usr/bin/env python3
"""Compare raw PyTorch and ONNX Runtime CUDA outputs for the same input."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    return parser.parse_args()


def pytorch_predict(checkpoint: Path, input_array: np.ndarray) -> np.ndarray:
    """Run the fused Ultralytics checkpoint on CUDA and return raw predictions."""
    model = YOLO(str(checkpoint)).model

    # Ultralytics fuses Conv and BatchNorm layers during ONNX export. Apply the
    # same inference optimization here so both backends execute equivalent graphs.
    model.fuse(verbose=False)
    model = model.eval().cuda()
    tensor = torch.from_numpy(input_array).cuda()

    with torch.inference_mode():
        output = model(tensor)

    # Ultralytics detection models return (predictions, feature_maps) outside
    # export mode. Only predictions correspond to the ONNX graph output.
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output.float().cpu().numpy()


def onnx_predict(model_path: Path, input_array: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Run ONNX with CUDA first and CPU available only as an operator fallback."""
    requested_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=requested_providers)
    active_providers = session.get_providers()
    if not active_providers or active_providers[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"ONNX Runtime did not activate CUDA first: {active_providers}")

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(f"Expected one input/output, found {len(inputs)}/{len(outputs)}")

    prediction = session.run([outputs[0].name], {inputs[0].name: input_array})[0]
    return prediction, active_providers


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    onnx_path = args.onnx.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    # A deterministic synthetic image isolates backend differences from image
    # decoding and preprocessing differences.
    rng = np.random.default_rng(seed=0)
    input_array = rng.random(
        (1, 3, args.imgsz, args.imgsz), dtype=np.float32
    )

    torch_output = pytorch_predict(checkpoint, input_array)
    onnx_output, providers = onnx_predict(onnx_path, input_array)

    if torch_output.shape != onnx_output.shape:
        raise ValueError(
            f"Output shape mismatch: PyTorch {torch_output.shape}, ONNX {onnx_output.shape}"
        )
    if not np.isfinite(torch_output).all() or not np.isfinite(onnx_output).all():
        raise ValueError("A backend produced NaN or infinite output values")

    absolute_error = np.abs(torch_output - onnx_output)
    close = np.allclose(
        torch_output, onnx_output, atol=args.atol, rtol=args.rtol
    )

    print(f"Providers: {providers}")
    print(f"Output shape: {torch_output.shape}")
    print(f"Maximum absolute error: {absolute_error.max():.8f}")
    print(f"Mean absolute error: {absolute_error.mean():.8f}")
    print(f"Tolerance: atol={args.atol}, rtol={args.rtol}")
    if not close:
        raise AssertionError("PyTorch and ONNX outputs exceed parity tolerance")
    print("Numerical parity: PASS")


if __name__ == "__main__":
    main()
