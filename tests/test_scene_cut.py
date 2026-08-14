from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rvfi_sr.scene_cut import detect_scene_cuts, parse_scdet_metadata


def make_metadata(scores: tuple[float, ...]) -> str:
    lines: list[str] = []
    for index, score in enumerate(scores):
        lines.extend(
            (
                f"frame:{index} pts:{index * 1000} pts_time:{index / 30:.6f}",
                f"lavfi.scd.mafd={score:.3f}",
                f"lavfi.scd.score={score:.3f}",
            )
        )
    return "\n".join(lines)


class SceneCutMetadataTest(unittest.TestCase):
    def test_threshold_maps_current_frame_to_previous_transition(self) -> None:
        result = parse_scdet_metadata(
            make_metadata((0.0, 3.0, 30.0, 26.999)),
            expected_frames=4,
            threshold=27.0,
        )
        self.assertEqual(result.cut_after, (1,))
        self.assertEqual(result.scores, (0.0, 3.0, 30.0, 26.999))

    def test_missing_or_nonsequential_frame_metadata_fails_fast(self) -> None:
        payload = make_metadata((0.0, 3.0, 4.0)).replace("frame:1", "frame:2")
        with self.assertRaisesRegex(ValueError, "sequential"):
            parse_scdet_metadata(payload, expected_frames=3, threshold=27.0)
        with self.assertRaisesRegex(ValueError, "expected 4"):
            parse_scdet_metadata(
                make_metadata((0.0, 3.0, 4.0)),
                expected_frames=4,
                threshold=27.0,
            )

    def test_invalid_score_and_threshold_fail_fast(self) -> None:
        payload = make_metadata((0.0, 3.0)).replace(
            "lavfi.scd.score=3.000",
            "lavfi.scd.score=nan",
        )
        with self.assertRaisesRegex(ValueError, "score"):
            parse_scdet_metadata(payload, expected_frames=2, threshold=27.0)
        with self.assertRaisesRegex(ValueError, "threshold"):
            parse_scdet_metadata(make_metadata((0.0, 3.0)), expected_frames=2, threshold=0.0)

    def test_ffmpeg_boundary_uses_metadata_stdout_and_fixed_filter(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            root = Path(temporary_directory)
            ffmpeg_path = root / "ffmpeg"
            input_path = root / "input.mp4"
            ffmpeg_path.touch(mode=0o755)
            input_path.touch()
            with patch("rvfi_sr.scene_cut.subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess(
                    args=(),
                    returncode=0,
                    stdout=make_metadata((0.0, 30.0)),
                    stderr="",
                )
                result = detect_scene_cuts(
                    ffmpeg_path,
                    input_path,
                    expected_frames=2,
                    threshold=27.0,
                )
            self.assertEqual(result.cut_after, (0,))
            command = run.call_args.args[0]
            self.assertIn("scdet=threshold=100,metadata=mode=print:file=-", command)
            self.assertEqual(command[-1], "-")
            self.assertTrue(run.call_args.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
