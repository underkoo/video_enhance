"""고정 FFmpeg scdet 점수를 scene transition index로 변환합니다."""

from __future__ import annotations

import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_FRAME_PATTERN = re.compile(r"^frame:(\d+)\s+pts:")
_SCORE_PREFIX = "lavfi.scd.score="


@dataclass(frozen=True, slots=True)
class SceneDetectionResult:
    """프레임별 score와 임계값을 넘은 transition을 함께 보존합니다."""

    threshold: float
    scores: tuple[float, ...]
    cut_after: tuple[int, ...]


def _validate_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, int | float):
        raise TypeError("threshold must be numeric")
    result = float(threshold)
    if not math.isfinite(result) or not 0.0 < result <= 100.0:
        raise ValueError("threshold must be finite and in (0, 100]")
    return result


def parse_scdet_metadata(
    payload: str,
    *,
    expected_frames: int,
    threshold: float,
) -> SceneDetectionResult:
    """metadata frame/score pair를 누락 없이 검증하고 `cut_after`로 변환합니다."""

    if not isinstance(payload, str):
        raise TypeError("payload must be a string")
    if isinstance(expected_frames, bool) or not isinstance(expected_frames, int):
        raise TypeError("expected_frames must be an integer")
    if expected_frames < 1:
        raise ValueError("expected_frames must be positive")
    validated_threshold = _validate_threshold(threshold)

    frame_indices: list[int] = []
    scores: list[float] = []
    pending_frame: int | None = None
    for line in payload.splitlines():
        frame_match = _FRAME_PATTERN.match(line)
        if frame_match is not None:
            if pending_frame is not None:
                raise ValueError(f"scene score missing for frame {pending_frame}")
            pending_frame = int(frame_match.group(1))
            continue
        if line.startswith(_SCORE_PREFIX):
            if pending_frame is None:
                raise ValueError("scene score has no preceding frame metadata")
            try:
                score = float(line.removeprefix(_SCORE_PREFIX))
            except ValueError as error:
                raise ValueError(f"invalid scene score for frame {pending_frame}") from error
            if not math.isfinite(score) or not 0.0 <= score <= 100.0:
                raise ValueError(f"scene score must be finite and in [0, 100]: {score!r}")
            frame_indices.append(pending_frame)
            scores.append(score)
            pending_frame = None

    if pending_frame is not None:
        raise ValueError(f"scene score missing for frame {pending_frame}")
    if len(scores) != expected_frames:
        raise ValueError(
            f"expected {expected_frames} scene scores, found {len(scores)}"
        )
    expected_indices = list(range(expected_frames))
    if frame_indices != expected_indices:
        raise ValueError("scene metadata frame indices must be sequential from zero")
    cut_after = tuple(
        frame_index - 1
        for frame_index, score in enumerate(scores)
        if frame_index > 0 and score >= validated_threshold
    )
    return SceneDetectionResult(
        threshold=validated_threshold,
        scores=tuple(scores),
        cut_after=cut_after,
    )


def detect_scene_cuts(
    ffmpeg_path: Path,
    input_path: Path,
    *,
    expected_frames: int,
    threshold: float,
) -> SceneDetectionResult:
    """모든 decoded frame의 pinned scdet score를 출력해 strict parser로 검증합니다."""

    if not isinstance(ffmpeg_path, Path) or not isinstance(input_path, Path):
        raise TypeError("ffmpeg_path and input_path must be pathlib.Path")
    executable = ffmpeg_path.resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise PermissionError(f"ffmpeg_path must be executable: {executable}")
    resolved_input = input_path.resolve(strict=True)
    if not resolved_input.is_file() or resolved_input.suffix.casefold() != ".mp4":
        raise ValueError("input_path must be an existing MP4 file")
    _validate_threshold(threshold)

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
        "scdet=threshold=100,metadata=mode=print:file=-",
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
        raise RuntimeError(f"FFmpeg scene detection failed: {stderr}") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("FFmpeg scene detection timed out") from error
    return parse_scdet_metadata(
        result.stdout,
        expected_frames=expected_frames,
        threshold=threshold,
    )
