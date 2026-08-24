from __future__ import annotations

import unittest

from training.coreml_spec import (
    CoreMLContractError,
    raw_yolo_output_contract,
    validate_raw_yolo_detection_contract,
)


def valid_spec() -> dict:
    return {
        "inputs": [
            {
                "name": "image",
                "type": "imageType",
                "width": 640,
                "height": 640,
            }
        ],
        "outputs": [
            {
                "name": "actual_exported_name",
                "type": "multiArrayType",
                "shape": [1, 5, 8400],
                "data_type": "FLOAT16",
            }
        ],
        "metadata": {"task": "detect", "end2end": "False"},
    }


class CoreMLContractTests(unittest.TestCase):
    def test_accepts_actual_seed_raw_output_shape_independent_of_output_name(self) -> None:
        validate_raw_yolo_detection_contract(
            valid_spec(),
            expected_input_size=640,
            expected_class_count=1,
        )

    def test_rejects_transposed_or_end_to_end_output(self) -> None:
        transposed = valid_spec()
        transposed["outputs"][0]["shape"] = [1, 8400, 5]
        with self.assertRaises(CoreMLContractError):
            validate_raw_yolo_detection_contract(
                transposed,
                expected_input_size=640,
                expected_class_count=1,
            )

    def test_documents_coreml_center_xywh_and_sdk_top_left_xywh(self) -> None:
        contract = raw_yolo_output_contract(1)
        self.assertEqual(
            contract["coreml_coordinate_channels"],
            ["center_x", "center_y", "width", "height"],
        )
        self.assertEqual(
            contract["sdk_coordinate_layout"],
            ["top_left_x", "top_left_y", "width", "height"],
        )
        self.assertEqual(contract["sdk_origin"], "top_left")


if __name__ == "__main__":
    unittest.main()
