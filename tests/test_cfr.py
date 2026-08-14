from __future__ import annotations

import subprocess
import tempfile
import unittest
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from rvfi_sr.cfr import (
    CfrPlan,
    cfr_filter_expression,
    parse_fps_filter_summary,
    probe_cfr_plan,
    stream_cfr_rgb24_frames,
)
from rvfi_sr.color import RgbDecodeContract, VideoColorRange, VideoColorSpace
from rvfi_sr.probe import ColorMetadata


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

    def test_rgb_stream_delivers_exact_frame_boundaries(self) -> None:
        class FakeProcess:
            def __init__(self, payload: bytes) -> None:
                self.stdout = BytesIO(payload)
                self.returncode = 0

            def wait(self, timeout: int | None = None) -> int:
                return self.returncode

            def poll(self) -> int:
                return self.returncode

            def terminate(self) -> None:
                self.returncode = -15

            def kill(self) -> None:
                self.returncode = -9

        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            root = Path(temporary_directory)
            ffmpeg_path = root / "ffmpeg"
            input_path = root / "input.mp4"
            ffmpeg_path.touch(mode=0o755)
            input_path.touch()
            plan = CfrPlan(2, 2, 0, 0, Fraction(30, 1))
            color = RgbDecodeContract.create(
                ColorMetadata("yuv420p", None, None, None, None),
                untagged_range=VideoColorRange.TV,
                untagged_space=VideoColorSpace.BT709,
            )
            received: list[tuple[int, bytes]] = []
            with patch(
                "rvfi_sr.cfr.subprocess.Popen",
                return_value=FakeProcess(bytes(range(12))),
            ) as popen:
                stream_cfr_rgb24_frames(
                    ffmpeg_path,
                    input_path,
                    width=2,
                    height=1,
                    plan=plan,
                    color=color,
                    consume_frame=lambda index, frame: received.append((index, frame)),
                )
            self.assertEqual(received, [(0, bytes(range(6))), (1, bytes(range(6, 12)))])
            command = popen.call_args.args[0]
            filter_expression = command[command.index("-vf") + 1]
            self.assertTrue(filter_expression.endswith(color.ffmpeg_filter))

    def test_rgb_stream_rejects_short_or_extra_payload(self) -> None:
        class FakeProcess:
            def __init__(self, payload: bytes) -> None:
                self.stdout = BytesIO(payload)
                self.returncode = 0

            def wait(self, timeout: int | None = None) -> int:
                return self.returncode

            def poll(self) -> int:
                return self.returncode

            def terminate(self) -> None:
                self.returncode = -15

            def kill(self) -> None:
                self.returncode = -9

        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            root = Path(temporary_directory)
            ffmpeg_path = root / "ffmpeg"
            input_path = root / "input.mp4"
            ffmpeg_path.touch(mode=0o755)
            input_path.touch()
            plan = CfrPlan(1, 1, 0, 0, Fraction(30, 1))
            color = RgbDecodeContract.create(
                ColorMetadata("yuv420p", None, None, None, None),
                untagged_range=VideoColorRange.TV,
                untagged_space=VideoColorSpace.BT709,
            )
            with patch(
                "rvfi_sr.cfr.subprocess.Popen",
                return_value=FakeProcess(b"short"),
            ), self.assertRaisesRegex(RuntimeError, "ended early"):
                stream_cfr_rgb24_frames(
                    ffmpeg_path,
                    input_path,
                    width=2,
                    height=1,
                    plan=plan,
                    color=color,
                    consume_frame=lambda _index, _frame: None,
                )
            with patch(
                "rvfi_sr.cfr.subprocess.Popen",
                return_value=FakeProcess(b"1234567"),
            ), self.assertRaisesRegex(RuntimeError, "exceeded"):
                stream_cfr_rgb24_frames(
                    ffmpeg_path,
                    input_path,
                    width=2,
                    height=1,
                    plan=plan,
                    color=color,
                    consume_frame=lambda _index, _frame: None,
                )


if __name__ == "__main__":
    unittest.main()
