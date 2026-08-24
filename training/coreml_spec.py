"""Read and validate the generated GolfBall Core ML package contract."""
from __future__ import annotations

from pathlib import Path
from typing import Any


class CoreMLContractError(RuntimeError):
    """Raised when a generated model cannot be consumed by the pinned iOS decoder."""


def _enum_name(enum_type: Any, value: int) -> str:
    try:
        return str(enum_type.Name(value))
    except (AttributeError, ValueError):
        return str(value)


def _feature_description(feature: Any, feature_types: Any) -> dict[str, Any]:
    kind = feature.type.WhichOneof("Type")
    result: dict[str, Any] = {
        "name": feature.name,
        "type": kind,
        "description": feature.shortDescription or None,
    }
    if kind == "imageType":
        image = feature.type.imageType
        result.update(
            {
                "width": int(image.width),
                "height": int(image.height),
                "color_space": _enum_name(
                    feature_types.ImageFeatureType.ColorSpace,
                    image.colorSpace,
                ),
            }
        )
    elif kind == "multiArrayType":
        array = feature.type.multiArrayType
        result.update(
            {
                "shape": [int(value) for value in array.shape],
                "data_type": _enum_name(
                    feature_types.ArrayFeatureType.ArrayDataType,
                    array.dataType,
                ),
            }
        )
    return result


def inspect_coreml_package(package: Path) -> dict[str, Any]:
    """Return the I/O and metadata declared by an actual .mlpackage specification."""
    if not package.is_dir():
        raise CoreMLContractError(f"Core ML package is missing: {package}")

    try:
        import coremltools as ct
        from coremltools.proto import FeatureTypes_pb2
    except ImportError as error:  # pragma: no cover - exercised by hosted model workflow
        raise CoreMLContractError("coremltools is required to inspect GolfBall.mlpackage") from error

    model = ct.models.MLModel(str(package), skip_model_load=True)
    spec = model.get_spec()
    description = spec.description
    return {
        "specification_version": int(spec.specificationVersion),
        "model_type": spec.WhichOneof("Type"),
        "inputs": [
            _feature_description(feature, FeatureTypes_pb2)
            for feature in description.input
        ],
        "outputs": [
            _feature_description(feature, FeatureTypes_pb2)
            for feature in description.output
        ],
        "metadata": dict(description.metadata.userDefined),
    }


def validate_raw_yolo_detection_contract(
    spec: dict[str, Any],
    *,
    expected_input_size: int,
    expected_class_count: int,
) -> None:
    """Validate the raw legacy YOLO tensor consumed by UltralyticsYOLO v8.9.13."""
    image_inputs = [item for item in spec["inputs"] if item["type"] == "imageType"]
    if len(image_inputs) != 1:
        raise CoreMLContractError(f"Expected one image input, got: {spec['inputs']}")
    image = image_inputs[0]
    if image["name"] != "image" or [image["width"], image["height"]] != [
        expected_input_size,
        expected_input_size,
    ]:
        raise CoreMLContractError(f"Unexpected Core ML image input: {image}")

    outputs = spec["outputs"]
    if len(outputs) != 1 or outputs[0]["type"] != "multiArrayType":
        raise CoreMLContractError(f"Expected one raw MLMultiArray output, got: {outputs}")
    shape = outputs[0].get("shape")
    expected_features = 4 + expected_class_count
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or shape[0] != 1
        or shape[1] != expected_features
        or shape[2] <= 0
    ):
        raise CoreMLContractError(
            "Expected raw YOLO [batch, 4 + classes, anchors] output "
            f"with {expected_class_count} class(es), got: {outputs[0]}"
        )

    metadata = spec["metadata"]
    if metadata.get("task") != "detect":
        raise CoreMLContractError(f"Core ML metadata task must be detect: {metadata}")
    if metadata.get("end2end", "False").lower() == "true":
        raise CoreMLContractError("Expected legacy raw YOLO output, not an end-to-end output")


def raw_yolo_output_contract(class_count: int) -> dict[str, Any]:
    """Describe semantics proven by the exporter and pinned SDK around the spec shape."""
    return {
        "coreml_tensor_layout": "[batch, 4 + class_count, anchors]",
        "coreml_coordinate_channels": ["center_x", "center_y", "width", "height"],
        "coreml_coordinate_units": "model_input_pixels",
        "coreml_confidence_channels": f"class_confidence[{class_count}] after coordinate channels",
        "coreml_objectness_channel": False,
        "sdk_output": "Box.xywhn",
        "sdk_coordinate_layout": ["top_left_x", "top_left_y", "width", "height"],
        "sdk_coordinate_units": "detector_input_normalized",
        "sdk_origin": "top_left",
    }
