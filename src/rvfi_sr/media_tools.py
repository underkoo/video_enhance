"""고정 FFmpeg toolchain을 안전한 subprocess 경계로 호출합니다."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, cast

from rvfi_sr.probe import MediaSpec, parse_ffprobe_payload


def _executable_file(path: Path, name: str) -> Path:
    if not isinstance(path, Path):
        raise TypeError(f"{name} must be pathlib.Path")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise PermissionError(f"{name} must be an executable file: {resolved}")
    return resolved


def run_ffprobe(ffprobe_path: Path, input_path: Path) -> MediaSpec:
    """frame count를 실제 decode하며 단일 MP4의 엄격한 명세를 반환합니다."""

    executable = _executable_file(ffprobe_path, "ffprobe_path")
    if not isinstance(input_path, Path):
        raise TypeError("input_path must be pathlib.Path")
    resolved_input = input_path.resolve(strict=True)
    if not resolved_input.is_file():
        raise FileNotFoundError(resolved_input)
    if resolved_input.suffix.casefold() != ".mp4":
        raise ValueError("input_path must use the .mp4 suffix")

    command = (
        str(executable),
        "-v",
        "error",
        "-count_frames",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(resolved_input),
    )
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.strip() if isinstance(error.stderr, str) else ""
        raise RuntimeError(f"ffprobe failed for {resolved_input}: {stderr}") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"ffprobe timed out for {resolved_input}") from error

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("ffprobe stdout must be valid JSON") from error
    if not isinstance(payload, dict):
        raise TypeError("ffprobe JSON root must be an object")
    return parse_ffprobe_payload(cast(dict[str, Any], payload))
