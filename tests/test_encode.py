from __future__ import annotations

import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from test_probe import make_payload

from rvfi_sr.encode import AtomicMp4Encoder, EncodeContract
from rvfi_sr.probe import parse_ffprobe_payload


def encoded_payload(*, audio: bool) -> dict[str, object]:
    payload = make_payload()
    video = payload["streams"][0]
    video["width"] = 2
    video["height"] = 2
    video["avg_frame_rate"] = "30/1"
    video["r_frame_rate"] = "30/1"
    video["nb_read_frames"] = "2"
    video["duration"] = "0.066667"
    video["color_range"] = "tv"
    video["color_space"] = "bt709"
    video["color_transfer"] = "bt709"
    video["color_primaries"] = "bt709"
    payload["format"]["duration"] = "0.066667"
    if audio:
        payload["streams"][1]["duration"] = "0.066667"
    if not audio:
        payload["streams"] = [video]
    return payload


class EncodeContractTest(unittest.TestCase):
    def test_command_explicitly_encodes_bt709_and_optional_audio(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.mp4"
            output_path = root / "output.mp4"
            ffmpeg_path = root / "ffmpeg"
            input_path.touch()
            ffmpeg_path.touch(mode=0o755)
            contract = EncodeContract.create(
                input_path=input_path,
                output_path=output_path,
                final_output_root=root,
                width=2,
                height=2,
                fps=Fraction(30, 1),
                frame_count=2,
                expect_audio=True,
            )
            command = contract.ffmpeg_command(ffmpeg_path)
            self.assertIn("libx264", command)
            self.assertIn("bt709", command)
            self.assertIn(
                "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited",
                command,
            )
            self.assertIn("1:a:0", command)
            self.assertIn("copy", command)
            self.assertLess(command.index(str(input_path)), command.index("1:a:0"))
            self.assertEqual(command[-1], str(contract.artifact.partial_path))

    def test_validation_rejects_frame_or_color_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.mp4"
            input_path.touch()
            contract = EncodeContract.create(
                input_path=input_path,
                output_path=root / "output.mp4",
                final_output_root=root,
                width=2,
                height=2,
                fps=Fraction(30, 1),
                frame_count=2,
                expect_audio=False,
            )
            contract.validate_output(parse_ffprobe_payload(encoded_payload(audio=False)))
            payload = encoded_payload(audio=False)
            payload["streams"][0]["nb_read_frames"] = "1"
            with self.assertRaisesRegex(ValueError, "frame count"):
                contract.validate_output(parse_ffprobe_payload(payload))
            payload = encoded_payload(audio=False)
            payload["streams"][0]["color_space"] = "smpte170m"
            with self.assertRaisesRegex(ValueError, "color metadata"):
                contract.validate_output(parse_ffprobe_payload(payload))

    def test_audio_duration_must_match_within_one_video_or_aac_frame(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.mp4"
            input_path.touch()
            contract = EncodeContract.create(
                input_path=input_path,
                output_path=root / "output.mp4",
                final_output_root=root,
                width=2,
                height=2,
                fps=Fraction(30, 1),
                frame_count=2,
                expect_audio=True,
            )
            contract.validate_output(parse_ffprobe_payload(encoded_payload(audio=True)))
            payload = encoded_payload(audio=True)
            payload["streams"][1]["duration"] = "0.088000"
            contract.validate_output(parse_ffprobe_payload(payload))
            payload = encoded_payload(audio=True)
            payload["streams"][1]["duration"] = "0.200000"
            with self.assertRaisesRegex(ValueError, "audio duration"):
                contract.validate_output(parse_ffprobe_payload(payload))

    def test_output_must_be_under_root_and_use_even_geometry(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.mp4"
            input_path.touch()
            with self.assertRaisesRegex(ValueError, "final_output_root"):
                EncodeContract.create(
                    input_path=input_path,
                    output_path=root.parent / "outside.mp4",
                    final_output_root=root,
                    width=2,
                    height=2,
                    fps=Fraction(30, 1),
                    frame_count=2,
                    expect_audio=False,
                )
            with self.assertRaisesRegex(ValueError, "even"):
                EncodeContract.create(
                    input_path=input_path,
                    output_path=root / "output.mp4",
                    final_output_root=root,
                    width=3,
                    height=2,
                    fps=Fraction(30, 1),
                    frame_count=2,
                    expect_audio=False,
                )

    def test_atomic_encoder_renames_only_after_probe_validation(self) -> None:
        class Sink:
            def __init__(self) -> None:
                self.payload = bytearray()

            def write(self, value: bytes) -> int:
                self.payload.extend(value)
                return len(value)

            def close(self) -> None:
                return None

        class FakeProcess:
            def __init__(self, partial_path: Path) -> None:
                self.stdin = Sink()
                self._partial_path = partial_path
                self.returncode = 0

            def wait(self, timeout: int | None = None) -> int:
                self._partial_path.write_bytes(b"encoded")
                return self.returncode

            def poll(self) -> int:
                return self.returncode

            def terminate(self) -> None:
                self.returncode = -15

            def kill(self) -> None:
                self.returncode = -9

        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.mp4"
            output_path = root / "output.mp4"
            ffmpeg_path = root / "ffmpeg"
            ffprobe_path = root / "ffprobe"
            input_path.touch()
            ffmpeg_path.touch(mode=0o755)
            ffprobe_path.touch(mode=0o755)
            contract = EncodeContract.create(
                input_path=input_path,
                output_path=output_path,
                final_output_root=root,
                width=2,
                height=2,
                fps=Fraction(30, 1),
                frame_count=2,
                expect_audio=False,
            )

            def fake_popen(command: tuple[str, ...], **_kwargs: object) -> FakeProcess:
                return FakeProcess(Path(command[-1]))

            with patch("rvfi_sr.encode.subprocess.Popen", side_effect=fake_popen), patch(
                "rvfi_sr.encode.run_ffprobe",
                return_value=parse_ffprobe_payload(encoded_payload(audio=False)),
            ):
                encoder = AtomicMp4Encoder(
                    contract,
                    ffmpeg_path=ffmpeg_path,
                    timeout_seconds=60,
                )
                encoder.consume(0, bytes(12))
                encoder.consume(1, bytes([1] * 12))
                result = encoder.finalize(ffprobe_path)
            self.assertEqual(result, output_path)
            self.assertEqual(output_path.read_bytes(), b"encoded")
            self.assertFalse(contract.artifact.partial_path.exists())


if __name__ == "__main__":
    unittest.main()
