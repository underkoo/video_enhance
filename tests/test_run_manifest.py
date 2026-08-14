from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rvfi_sr.run_manifest import (
    RunPlanManifest,
    validate_run_plan,
    write_completed_run_manifest,
    write_run_plan,
)


def make_plan(root: Path) -> RunPlanManifest:
    return RunPlanManifest(
        schema_version=2,
        input_path=str(root / "input.mp4"),
        output_path=str(root / "output.mp4"),
        input_sha256="a" * 64,
        config_sha256="b" * 64,
        order="vfi_then_vsr",
        source_width=604,
        source_height=1080,
        source_frames=190,
        cfr_frames=191,
        cfr_dropped_frames=0,
        cfr_duplicated_frames=1,
        target_fps="30/1",
        scene_cut_after=(),
        vfi_backend_id="practical-rife-v4.25",
        vfi_multiplier=2,
        rife_chunk_count=4,
        vsr_backend_id="mmagic-realbasicvsr",
        vsr_scale=2,
        vsr_max_model_frames=6,
        vsr_chunk_count=95,
        output_fps="60/1",
        output_frames=382,
    )


class RunManifestTest(unittest.TestCase):
    def test_plan_round_trip_and_field_level_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            root = Path(temporary_directory)
            path = root / "run-plan.json"
            plan = make_plan(root)
            write_run_plan(path, plan)
            validate_run_plan(path, plan)
            with self.assertRaisesRegex(RuntimeError, "config_sha256"):
                validate_run_plan(path, replace(plan, config_sha256="c" * 64))
            with self.assertRaises(FileExistsError):
                write_run_plan(path, plan)

    def test_completed_manifest_contains_plan_and_verified_result(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            root = Path(temporary_directory)
            output = root / "output.run.json"
            write_completed_run_manifest(
                output,
                plan=make_plan(root),
                output_sha256="d" * 64,
                output_size_bytes=1234,
                elapsed_seconds=12.5,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["plan"]["cfr_frames"], 191)
            self.assertEqual(payload["result"]["output_size_bytes"], 1234)
            self.assertTrue(output.read_bytes().endswith(b"\n"))
