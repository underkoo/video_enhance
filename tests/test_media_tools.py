from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_probe import make_payload

from rvfi_sr.media_tools import run_ffprobe


class MediaToolsTest(unittest.TestCase):
    def test_ffprobe_command_is_noninteractive_and_counted(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            root = Path(temporary_directory)
            ffprobe_path = root / "ffprobe"
            input_path = root / "input.mp4"
            ffprobe_path.touch(mode=0o755)
            input_path.touch()
            with patch("rvfi_sr.media_tools.subprocess.run") as run:
                run.return_value.stdout = json.dumps(make_payload())
                run.return_value.stderr = ""
                run.return_value.returncode = 0
                spec = run_ffprobe(ffprobe_path, input_path)
            self.assertEqual(spec.frame_count, 300)
            command = run.call_args.args[0]
            self.assertIn("-count_frames", command)
            self.assertIn("-show_streams", command)
            self.assertIn("-show_format", command)
            self.assertEqual(command[-1], str(input_path.resolve()))
            self.assertTrue(run.call_args.kwargs["check"])

    def test_missing_binary_and_invalid_json_fail_fast(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.mp4"
            input_path.touch()
            with self.assertRaises(FileNotFoundError):
                run_ffprobe(root / "missing-ffprobe", input_path)

            ffprobe_path = root / "ffprobe"
            ffprobe_path.touch(mode=0o755)
            with patch("rvfi_sr.media_tools.subprocess.run") as run:
                run.return_value.stdout = "not-json"
                run.return_value.stderr = ""
                run.return_value.returncode = 0
                with self.assertRaisesRegex(ValueError, "valid JSON"):
                    run_ffprobe(ffprobe_path, input_path)


if __name__ == "__main__":
    unittest.main()
