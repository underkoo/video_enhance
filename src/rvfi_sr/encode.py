"""RGB24 stream을 검증 후 atomic 확정하는 MP4 encoder입니다."""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO, Self, cast

from rvfi_sr.artifacts import ArtifactContract
from rvfi_sr.media_tools import run_ffprobe
from rvfi_sr.probe import MediaSpec


@dataclass(frozen=True, slots=True)
class EncodeContract:
    """최종 H.264 MP4의 경로·timeline·color·audio 기대값입니다."""

    artifact: ArtifactContract
    final_output_root: Path
    width: int
    height: int
    fps: Fraction
    frame_count: int
    expect_audio: bool
    source_audio_duration: Fraction | None
    crf: int
    preset: str

    @classmethod
    def create(
        cls,
        *,
        input_path: Path,
        output_path: Path,
        final_output_root: Path,
        width: int,
        height: int,
        fps: Fraction,
        frame_count: int,
        expect_audio: bool,
        source_audio_duration: Fraction | None = None,
        crf: int = 16,
        preset: str = "slow",
    ) -> Self:
        """출력 생성 전에 경로와 encoder 제약을 모두 검증합니다."""

        if not isinstance(final_output_root, Path) or not final_output_root.is_absolute():
            raise ValueError("final_output_root must be an absolute pathlib.Path")
        resolved_root = final_output_root.resolve(strict=False)
        artifact = ArtifactContract.create(input_path, output_path)
        if not (
            artifact.output_path == resolved_root
            or artifact.output_path.is_relative_to(resolved_root)
        ):
            raise ValueError(f"output_path must be under final_output_root: {resolved_root}")
        if artifact.output_path.exists() or artifact.partial_path.exists():
            raise FileExistsError(
                artifact.output_path
                if artifact.output_path.exists()
                else artifact.partial_path
            )
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or isinstance(height, bool)
            or not isinstance(height, int)
        ):
            raise TypeError("width and height must be integers")
        if width < 2 or height < 2 or width % 2 or height % 2:
            raise ValueError("yuv420p output width and height must be positive even values")
        if not isinstance(fps, Fraction):
            raise TypeError("fps must be fractions.Fraction")
        if fps <= 0:
            raise ValueError("fps must be positive")
        if isinstance(frame_count, bool) or not isinstance(frame_count, int):
            raise TypeError("frame_count must be an integer")
        if frame_count < 1:
            raise ValueError("frame_count must be positive")
        if not isinstance(expect_audio, bool):
            raise TypeError("expect_audio must be boolean")
        if expect_audio:
            if (
                not isinstance(source_audio_duration, Fraction)
                or source_audio_duration <= 0
            ):
                raise ValueError(
                    "source_audio_duration must be a positive Fraction when audio is expected"
                )
        elif source_audio_duration is not None:
            raise ValueError("source_audio_duration must be None when audio is not expected")
        if isinstance(crf, bool) or not isinstance(crf, int) or not 0 <= crf <= 51:
            raise ValueError("crf must be an integer in [0, 51]")
        if preset not in {"medium", "slow", "slower"}:
            raise ValueError("preset must be one of medium, slow, slower")
        return cls(
            artifact=artifact,
            final_output_root=resolved_root,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            expect_audio=expect_audio,
            source_audio_duration=source_audio_duration,
            crf=crf,
            preset=preset,
        )

    @property
    def expected_duration(self) -> Fraction:
        return Fraction(self.frame_count, 1) / self.fps

    def validate_output(self, spec: MediaSpec) -> None:
        """atomic rename 전에 ffprobe 결과가 전체 계약과 일치하는지 검사합니다."""

        if not isinstance(spec, MediaSpec):
            raise TypeError("spec must be MediaSpec")
        if spec.video_codec != "h264":
            raise ValueError(f"output video codec must be h264, got {spec.video_codec!r}")
        if (spec.width, spec.height) != (self.width, self.height):
            raise ValueError(
                f"output dimensions mismatch: expected={(self.width, self.height)}, "
                f"actual={(spec.width, spec.height)}"
            )
        if spec.frame_count != self.frame_count:
            raise ValueError(
                f"output frame count mismatch: expected={self.frame_count}, "
                f"actual={spec.frame_count}"
            )
        if spec.average_fps != self.fps or spec.nominal_fps != self.fps:
            raise ValueError(
                f"output FPS mismatch: expected={self.fps}, "
                f"average={spec.average_fps}, nominal={spec.nominal_fps}"
            )
        tolerance = Fraction(1, 1) / self.fps
        if abs(spec.video_duration - self.expected_duration) > tolerance:
            raise ValueError(
                f"output video duration mismatch: expected={self.expected_duration}, "
                f"actual={spec.video_duration}, tolerance={tolerance}"
            )
        if spec.color.pixel_format != "yuv420p":
            raise ValueError("output pixel format must be yuv420p")
        expected_color = ("tv", "bt709", "bt709", "bt709")
        actual_color = (
            spec.color.range,
            spec.color.space,
            spec.color.transfer,
            spec.color.primaries,
        )
        if actual_color != expected_color:
            raise ValueError(
                f"output color metadata mismatch: expected={expected_color}, actual={actual_color}"
            )
        if (spec.audio is not None) != self.expect_audio:
            raise ValueError(
                f"output audio presence mismatch: expected={self.expect_audio}, "
                f"actual={spec.audio is not None}"
            )
        if spec.audio is not None:
            if spec.audio.codec != "aac":
                raise ValueError(f"output audio codec must be aac, got {spec.audio.codec!r}")
            assert self.source_audio_duration is not None
            audio_tolerance = Fraction(1024, spec.audio.sample_rate)
            if abs(spec.audio.duration - self.source_audio_duration) > audio_tolerance:
                raise ValueError(
                    "remuxed audio duration mismatch: "
                    f"expected_source={self.source_audio_duration}, "
                    f"actual={spec.audio.duration}, tolerance={audio_tolerance}"
                )

    def ffmpeg_command(self, ffmpeg_path: Path) -> tuple[str, ...]:
        """고정 RGB→BT.709 limited encode와 optional audio remux command를 반환합니다."""

        executable = _executable(ffmpeg_path, "ffmpeg_path")
        rate = f"{self.fps.numerator}/{self.fps.denominator}"
        command = [
            str(executable),
            "-v",
            "error",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            rate,
            "-i",
            "pipe:0",
        ]
        if self.expect_audio:
            command.extend(
                (
                    "-i",
                    str(self.artifact.input_path),
                )
            )
        command.extend(("-map", "0:v:0"))
        if self.expect_audio:
            command.extend(("-map", "1:a:0"))
        command.extend(
            (
            "-vf",
            "scale=in_range=pc:out_range=tv:out_color_matrix=bt709,"
            "format=pix_fmts=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            self.preset,
            "-crf",
            str(self.crf),
            "-pix_fmt",
            "yuv420p",
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_primaries",
            "bt709",
            "-x264-params",
            "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited",
            "-fps_mode",
            "cfr",
            "-movflags",
            "+faststart",
            "-n",
            str(self.artifact.partial_path),
            )
        )
        if self.expect_audio:
            output_path = command.pop()
            command.extend(("-c:a", "copy", output_path))
        return tuple(command)


def _executable(path: Path, name: str) -> Path:
    if not isinstance(path, Path):
        raise TypeError(f"{name} must be pathlib.Path")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise PermissionError(f"{name} must be an executable file: {resolved}")
    return resolved


class AtomicMp4Encoder:
    """정확한 RGB frame stream만 받고 ffprobe 통과 후 결과를 atomic 확정합니다."""

    def __init__(
        self,
        contract: EncodeContract,
        *,
        ffmpeg_path: Path,
        timeout_seconds: int,
    ) -> None:
        if not isinstance(contract, EncodeContract):
            raise TypeError("contract must be EncodeContract")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or timeout_seconds < 60
        ):
            raise ValueError("timeout_seconds must be an integer >= 60")
        contract.artifact.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._contract = contract
        self._timeout_seconds = timeout_seconds
        self._error_stream = tempfile.TemporaryFile(mode="w+b")
        self._process = subprocess.Popen(
            contract.ffmpeg_command(ffmpeg_path),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._error_stream,
        )
        if self._process.stdin is None:
            self._process.kill()
            self._process.wait()
            self._error_stream.close()
            raise RuntimeError("FFmpeg stdin pipe was not created")
        self._stdin = cast(BinaryIO, self._process.stdin)
        self._frame_bytes = contract.width * contract.height * 3
        self._next_frame_index = 0
        self._closed = False

    def consume(self, frame_index: int, frame: bytes) -> None:
        """정확히 순차적인 RGB24 frame 하나를 encoder stdin에 기록합니다."""

        if self._closed:
            raise RuntimeError("encoder is already closed")
        if frame_index != self._next_frame_index:
            raise ValueError(
                f"frame index must be sequential: expected={self._next_frame_index}, "
                f"actual={frame_index}"
            )
        if not isinstance(frame, bytes):
            raise TypeError("frame must be bytes")
        if len(frame) != self._frame_bytes:
            raise ValueError(
                f"RGB24 frame byte count mismatch: expected={self._frame_bytes}, "
                f"actual={len(frame)}"
            )
        try:
            self._stdin.write(frame)
        except BrokenPipeError as error:
            diagnostic = self._diagnostic()
            raise RuntimeError(f"FFmpeg encoder pipe closed early: {diagnostic}") from error
        self._next_frame_index += 1

    def finalize(self, ffprobe_path: Path) -> Path:
        """frame count, process status, ffprobe 계약을 검증하고 atomic rename합니다."""

        if self._closed:
            raise RuntimeError("encoder is already closed")
        if self._next_frame_index != self._contract.frame_count:
            raise RuntimeError(
                f"encoder frame count mismatch: expected={self._contract.frame_count}, "
                f"actual={self._next_frame_index}"
            )
        self._stdin.close()
        return_code = self._process.wait(timeout=self._timeout_seconds)
        self._closed = True
        diagnostic = self._diagnostic()
        self._error_stream.close()
        if return_code != 0:
            raise RuntimeError(
                f"FFmpeg encode failed: returncode={return_code}, stderr={diagnostic}"
            )
        spec = run_ffprobe(ffprobe_path, self._contract.artifact.partial_path)
        self._contract.validate_output(spec)
        os.replace(
            self._contract.artifact.partial_path,
            self._contract.artifact.output_path,
        )
        return self._contract.artifact.output_path

    def abort(self) -> None:
        """현재 encoder와 이 인스턴스가 만든 partial artifact를 정리합니다."""

        if not self._closed:
            self._stdin.close()
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
            self._closed = True
            self._error_stream.close()
        self._contract.artifact.partial_path.unlink(missing_ok=True)

    def _diagnostic(self) -> str:
        self._error_stream.flush()
        self._error_stream.seek(0)
        return self._error_stream.read().decode("utf-8", errors="replace").strip()
