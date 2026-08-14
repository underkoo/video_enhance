"""고정 FFmpeg fps filter의 CFR 재타이밍 계획을 검증합니다."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO, cast

from rvfi_sr.color import RgbDecodeContract

_FPS_SUMMARY_PATTERN = re.compile(
    r"(?P<input>\d+) frames in, (?P<output>\d+) frames out; "
    r"(?P<dropped>\d+) frames dropped, (?P<duplicated>\d+) frames duplicated\."
)


@dataclass(frozen=True, slots=True)
class CfrPlan:
    """한 영상의 실제 fps filter 입출력 및 drop/dup 개수를 보존합니다."""

    source_frames: int
    output_frames: int
    dropped_frames: int
    duplicated_frames: int
    target_fps: Fraction

    def __post_init__(self) -> None:
        if self.source_frames < 1 or self.output_frames < 1:
            raise ValueError("CFR frame counts must be positive")
        if self.dropped_frames < 0 or self.duplicated_frames < 0:
            raise ValueError("CFR drop/dup counts must be non-negative")
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")
        expected_output = (
            self.source_frames - self.dropped_frames + self.duplicated_frames
        )
        if self.output_frames != expected_output:
            raise ValueError(
                "CFR accounting mismatch: "
                f"source={self.source_frames}, output={self.output_frames}, "
                f"dropped={self.dropped_frames}, duplicated={self.duplicated_frames}"
            )


def cfr_filter_expression(target_fps: Fraction) -> str:
    """재현 가능한 PTS origin, rounding, EOF 정책을 포함한 filter를 반환합니다."""

    if not isinstance(target_fps, Fraction):
        raise TypeError("target_fps must be fractions.Fraction")
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    rate = f"{target_fps.numerator}/{target_fps.denominator}"
    return f"setpts=PTS-STARTPTS,fps=fps={rate}:round=near:eof_action=pass"


def parse_fps_filter_summary(
    payload: str,
    *,
    expected_source_frames: int,
    target_fps: Fraction,
) -> CfrPlan:
    """FFmpeg verbose summary를 단 하나만 허용하고 frame accounting을 검증합니다."""

    if not isinstance(payload, str):
        raise TypeError("payload must be a string")
    if (
        isinstance(expected_source_frames, bool)
        or not isinstance(expected_source_frames, int)
    ):
        raise TypeError("expected_source_frames must be an integer")
    if expected_source_frames < 1:
        raise ValueError("expected_source_frames must be positive")
    cfr_filter_expression(target_fps)

    matches = tuple(_FPS_SUMMARY_PATTERN.finditer(payload))
    summaries = tuple(
        {name: int(value) for name, value in match.groupdict().items()}
        for match in matches
    )
    # FFmpeg 8 may configure and immediately dispose an empty probe graph before
    # running the real graph. Its all-zero summary carries no decoded timeline.
    meaningful = tuple(
        summary
        for summary in summaries
        if summary != {"input": 0, "output": 0, "dropped": 0, "duplicated": 0}
    )
    if len(meaningful) != 1:
        raise ValueError(
            f"expected exactly one non-empty fps filter summary, found {len(meaningful)}"
        )
    values = meaningful[0]
    if values["input"] != expected_source_frames:
        raise ValueError(
            "decoded source frame count mismatch: "
            f"expected={expected_source_frames}, actual={values['input']}"
        )
    return CfrPlan(
        source_frames=values["input"],
        output_frames=values["output"],
        dropped_frames=values["dropped"],
        duplicated_frames=values["duplicated"],
        target_fps=target_fps,
    )


def probe_cfr_plan(
    ffmpeg_path: Path,
    input_path: Path,
    *,
    expected_source_frames: int,
    target_fps: Fraction,
) -> CfrPlan:
    """동일 decode/filter chain을 null sink로 실행해 실제 CFR 계획을 확정합니다."""

    if not isinstance(ffmpeg_path, Path) or not isinstance(input_path, Path):
        raise TypeError("ffmpeg_path and input_path must be pathlib.Path")
    executable = ffmpeg_path.resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise PermissionError(f"ffmpeg_path must be executable: {executable}")
    resolved_input = input_path.resolve(strict=True)
    if not resolved_input.is_file() or resolved_input.suffix.casefold() != ".mp4":
        raise ValueError("input_path must be an existing MP4 file")
    filter_expression = cfr_filter_expression(target_fps)

    command = (
        str(executable),
        "-v",
        "verbose",
        "-nostdin",
        "-i",
        str(resolved_input),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        filter_expression,
        "-fps_mode",
        "passthrough",
        "-f",
        "null",
        "-",
    )
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.strip() if isinstance(error.stderr, str) else ""
        raise RuntimeError(f"FFmpeg CFR preflight failed: {stderr}") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("FFmpeg CFR preflight timed out") from error
    return parse_fps_filter_summary(
        result.stderr,
        expected_source_frames=expected_source_frames,
        target_fps=target_fps,
    )


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    payload = bytearray()
    while len(payload) < size:
        chunk = stream.read(size - len(payload))
        if not chunk:
            break
        payload.extend(chunk)
    return bytes(payload)


def stream_cfr_rgb24_frames(
    ffmpeg_path: Path,
    input_path: Path,
    *,
    width: int,
    height: int,
    plan: CfrPlan,
    color: RgbDecodeContract,
    consume_frame: Callable[[int, bytes], None],
) -> None:
    """검증된 CFR plan과 정확히 같은 수의 RGB24 frame을 callback으로 전달합니다."""

    if not isinstance(ffmpeg_path, Path) or not isinstance(input_path, Path):
        raise TypeError("ffmpeg_path and input_path must be pathlib.Path")
    executable = ffmpeg_path.resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise PermissionError(f"ffmpeg_path must be executable: {executable}")
    resolved_input = input_path.resolve(strict=True)
    if not resolved_input.is_file() or resolved_input.suffix.casefold() != ".mp4":
        raise ValueError("input_path must be an existing MP4 file")
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or isinstance(height, bool)
        or not isinstance(height, int)
    ):
        raise TypeError("width and height must be integers")
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    if not isinstance(plan, CfrPlan):
        raise TypeError("plan must be CfrPlan")
    if not isinstance(color, RgbDecodeContract):
        raise TypeError("color must be RgbDecodeContract")
    if not callable(consume_frame):
        raise TypeError("consume_frame must be callable")

    filter_expression = f"{cfr_filter_expression(plan.target_fps)},{color.ffmpeg_filter}"
    command = (
        str(executable),
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(resolved_input),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        filter_expression,
        "-fps_mode",
        "passthrough",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    )
    frame_bytes = width * height * 3
    with tempfile.TemporaryFile(mode="w+b") as error_stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=error_stream)
        if process.stdout is None:
            process.kill()
            process.wait()
            raise RuntimeError("FFmpeg stdout pipe was not created")
        stdout = cast(BinaryIO, process.stdout)
        completed = False
        try:
            for frame_index in range(plan.output_frames):
                frame = _read_exact(stdout, frame_bytes)
                if len(frame) != frame_bytes:
                    return_code = process.wait(timeout=30)
                    error_stream.seek(0)
                    diagnostic = error_stream.read().decode("utf-8", errors="replace").strip()
                    raise RuntimeError(
                        "FFmpeg RGB stream ended early: "
                        f"frame={frame_index}, bytes={len(frame)}/{frame_bytes}, "
                        f"returncode={return_code}, stderr={diagnostic}"
                    )
                consume_frame(frame_index, frame)
            if stdout.read(1):
                raise RuntimeError(
                    f"FFmpeg RGB stream exceeded expected {plan.output_frames} frames"
                )
            return_code = process.wait(timeout=30)
            error_stream.seek(0)
            diagnostic = error_stream.read().decode("utf-8", errors="replace").strip()
            if return_code != 0:
                raise RuntimeError(
                    f"FFmpeg RGB decode failed: returncode={return_code}, stderr={diagnostic}"
                )
            if diagnostic:
                raise RuntimeError(f"FFmpeg emitted an error-level diagnostic: {diagnostic}")
            completed = True
        finally:
            stdout.close()
            if not completed and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
