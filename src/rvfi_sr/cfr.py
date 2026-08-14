"""고정 FFmpeg fps filter의 CFR 재타이밍 계획을 검증합니다."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

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
