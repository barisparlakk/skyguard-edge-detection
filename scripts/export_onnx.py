#!/usr/bin/env python3
"""Export the trained Skyguard checkpoint to a validated ONNX opset-17 graph."""

from __future__ import annotations

import argparse
import functools
import inspect
import shutil
from pathlib import Path

import onnx
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the trained Ultralytics best.pt checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("export/skyguard_yolov8m_opset17.onnx"),
        help="Destination ONNX path",
    )
    return parser.parse_args()


def describe_value(value_info: onnx.ValueInfoProto) -> str:
    """Return a readable tensor name, element type, and shape."""
    tensor = value_info.type.tensor_type
    dimensions = []
    for dimension in tensor.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(str(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append("?")
    return f"{value_info.name}: dtype={tensor.elem_type}, shape=[{', '.join(dimensions)}]"


def validate_export(path: Path) -> onnx.ModelProto:
    """Validate graph structure, opset, and dynamic image input dimensions."""
    graph = onnx.load(path)
    onnx.checker.check_model(graph)

    default_domain_opsets = [
        item.version for item in graph.opset_import if item.domain in ("", "ai.onnx")
    ]
    if default_domain_opsets != [17]:
        raise ValueError(f"Expected ONNX opset 17, found {default_domain_opsets}")

    if len(graph.graph.input) != 1:
        raise ValueError(f"Expected one model input, found {len(graph.graph.input)}")

    input_shape = graph.graph.input[0].type.tensor_type.shape.dim
    if len(input_shape) != 4:
        raise ValueError(f"Expected NCHW rank-4 input, found rank {len(input_shape)}")
    if input_shape[1].dim_value != 3:
        raise ValueError("Expected three input channels")

    # Dynamic batch, height, and width let runtimes choose deployment profiles.
    for index, label in ((0, "batch"), (2, "height"), (3, "width")):
        if input_shape[index].HasField("dim_value"):
            raise ValueError(f"Expected dynamic {label}, found a fixed dimension")

    return graph


def apply_torch_onnx_compatibility() -> bool:
    """Remove only unsupported ONNX kwargs on the pinned NGC PyTorch build.

    Ultralytics 8.4 passes ``dynamo=False`` to ``torch.onnx.export``. NVIDIA's
    PyTorch 2.4 build in NGC 24.05 predates that keyword, although its legacy
    ONNX exporter is otherwise compatible with the requested operation.
    """
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        return False

    original_export = torch.onnx.export

    @functools.wraps(original_export)
    def compatible_export(*args, **kwargs):
        kwargs.pop("dynamo", None)
        return original_export(*args, **kwargs)

    torch.onnx.export = compatible_export
    return True


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    output_path = args.output.resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if apply_torch_onnx_compatibility():
        print("Applied NGC PyTorch 2.4 ONNX compatibility shim")

    # Export raw FP32 predictions without NMS. Post-processing must remain the
    # same across PyTorch, ONNX Runtime, and TensorRT benchmarks.
    model = YOLO(str(model_path))
    generated_path = Path(
        model.export(
            format="onnx",
            opset=17,
            imgsz=640,
            batch=1,
            dynamic=True,
            simplify=False,
            half=False,
            nms=False,
            device="cpu",
        )
    ).resolve()

    graph = validate_export(generated_path)
    shutil.move(str(generated_path), output_path)

    print(f"Validated ONNX export: {output_path}")
    print(f"IR version: {graph.ir_version}")
    print("Opset: 17")
    for value in graph.graph.input:
        print(f"Input  {describe_value(value)}")
    for value in graph.graph.output:
        print(f"Output {describe_value(value)}")


if __name__ == "__main__":
    main()
