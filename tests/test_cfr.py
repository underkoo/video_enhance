from __future__ import annotations

import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from rvfi_sr.cfr import (
    cfr_filter_expression,
    parse_fps_filter_summary,
    probe_cfr_plan,
)


def make_summary(
    source: int = 986,
    output: int = 984,
    dropped: int = 2,
    duplicated: int = 0,
) -> str:
    return (
        "[Parsed_fps_1 @ 0x123] "
        f"{source} frames in, {output} frames out; "
        f"{dropped} frames dropped, {duplicated} frames duplicated.\n"
    )


class CfrPlanTest(unittest.TestCase):
    def test_filter_and_summary_preserve_exact_rational_accounting(self) -> None:
        target_fps = Fraction(30_000, 1_001)
        self.assertEqual(
            cfr_filter_expression(target_fps),
            "setpts=PTS-STARTPTS,fps=fps=30000/1001:round=near:eof_action=pass",
        )
        plan = parse_fps_filter_summary(
            make_summary(source=0, output=0, dropped=0) + make_summary(),
            expected_source_frames=986,
            target_fps=target_fps,
        )
        self.assertEqual(plan.output_frames, 984)
        self.assertEqual(plan.dropped_frames, 2)
        self.assertEqual(plan.duplicated_frames, 0)

    def test_missing_duplicate_and_inconsistent_summaries_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            parse_fps_filter_summary(
                "no summary",
                expected_source_frames=986,
                target_fps=Fraction(30, 1),
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            parse_fps_filter_summary(
                make_summary() * 2,
                expected_source_frames=986,
                target_fps=Fraction(30, 1),
            )
        with self.assertRaisesRegex(ValueError, "accounting mismatch"):
            parse_fps_filter_summary(
                make_summary(output=985),
                expected_source_frames=986,
                target_fps=Fraction(30, 1),
            )

    def test_decoded_source_count_mismatch_fails_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "source frame count mismatch"):
            parse_fps_filter_summary(
                make_summary(),
                expected_source_frames=985,
                target_fps=Fraction(30, 1),
            )

    def test_ffmpeg_boundary_uses_identical_cfr_filter(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            root = Path(temporary_directory)
            ffmpeg_path = root / "ffmpeg"
            input_path = root / "input.mp4"
            ffmpeg_path.touch(mode=0o755)
            input_path.touch()
            with patch("rvfi_sr.cfr.subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess(
                    args=(),
                    returncode=0,
                    stdout="",
                    stderr=make_summary(),
                )
                plan = probe_cfr_plan(
                    ffmpeg_path,
                    input_path,
                    expected_source_frames=986,
                    target_fps=Fraction(30, 1),
                )
            self.assertEqual(plan.output_frames, 984)
            command = run.call_args.args[0]
            self.assertIn(cfr_filter_expression(Fraction(30, 1)), command)
            self.assertIn("passthrough", command)
            self.assertTrue(run.call_args.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
