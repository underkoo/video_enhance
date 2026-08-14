from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from rvfi_sr.config import PipelineConfig


def make_payload(root: Path) -> dict[str, object]:
    input_directory = root / "input"
    input_directory.mkdir()
    return {
        "input_dir": str(input_directory.resolve()),
        "output_dir": str((root / "output").resolve()),
        "vfi": {
            "backend_id": "practical-rife-v4.25",
            "temporal_multiplier": 2,
        },
        "vsr": {
            "backend_id": "flashvsr-v1.1",
            "spatial_scale": 2,
            "post_downsample": True,
        },
        "runtime": {
            "allow_restricted_license": False,
            "fp16": True,
            "gpu_index": 0,
        },
    }


class PipelineConfigTest(unittest.TestCase):
    def test_default_permissive_pipeline_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = PipelineConfig.model_validate(make_payload(Path(temporary_directory)))
            self.assertEqual(config.vfi.temporal_multiplier, 2)
            self.assertEqual(config.vsr.spatial_scale, 2)
            self.assertTrue(config.vsr.post_downsample)

    def test_unknown_fields_fail_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            payload = make_payload(Path(temporary_directory))
            payload["silent_fallback"] = True
            with self.assertRaisesRegex(ValidationError, "silent_fallback"):
                PipelineConfig.model_validate(payload)

    def test_restricted_backend_requires_runtime_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            payload = make_payload(Path(temporary_directory))
            payload["vfi"]["backend_id"] = "bim-vfi"
            with self.assertRaisesRegex(ValidationError, "restricted license"):
                PipelineConfig.model_validate(payload)
            payload["runtime"]["allow_restricted_license"] = True
            PipelineConfig.model_validate(payload)

    def test_output_must_not_overlap_input_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            payload = make_payload(root)
            payload["output_dir"] = str((root / "input" / "enhanced").resolve())
            with self.assertRaisesRegex(ValidationError, "must not overlap"):
                PipelineConfig.model_validate(payload)

    def test_backend_capability_mismatch_fails_during_config_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            payload = make_payload(Path(temporary_directory))
            payload["vfi"]["temporal_multiplier"] = 3
            with self.assertRaisesRegex(ValidationError, "temporal multiplier"):
                PipelineConfig.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
