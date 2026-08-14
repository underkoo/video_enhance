"""실제 MP4의 첫 21프레임으로 FlashVSR 해상도/VRAM smoke를 수행합니다."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np

from rvfi_sr.worker_protocol import WorkerRequest, WorkerResponse, WorkerStatus

_FRAME_COUNT = 21


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-scale", choices=(2, 4), default=2, type=int)
    return parser.parse_args()


def _probe_dimensions(ffprobe_path: Path, input_path: Path) -> tuple[int, int]:
    result = subprocess.run(
        (
            str(ffprobe_path),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(input_path),
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise TypeError("ffprobe payload root must be an object")
    streams = cast(dict[str, Any], payload).get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise ValueError("ffprobe must return exactly one selected video stream")
    stream = streams[0]
    if not isinstance(stream, dict):
        raise TypeError("ffprobe stream must be an object")
    width = stream.get("width")
    height = stream.get("height")
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or width < 1
        or not isinstance(height, int)
        or isinstance(height, bool)
        or height < 1
    ):
        raise ValueError("ffprobe width and height must be positive integers")
    return width, height


def _decode_first_chunk(
    ffmpeg_path: Path,
    input_path: Path,
    *,
    width: int,
    height: int,
) -> np.ndarray[Any, np.dtype[np.uint8]]:
    result = subprocess.run(
        (
            str(ffmpeg_path),
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-frames:v",
            str(_FRAME_COUNT),
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ),
        check=True,
        capture_output=True,
        timeout=300,
    )
    expected_bytes = _FRAME_COUNT * height * width * 3
    if len(result.stdout) != expected_bytes:
        raise RuntimeError(
            f"decoded byte count mismatch: expected={expected_bytes}, actual={len(result.stdout)}"
        )
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(
        _FRAME_COUNT,
        height,
        width,
        3,
    )


def main() -> int:
    """실제 해상도 chunk를 decode하고 격리 worker를 한 번 실행합니다."""

    arguments = _arguments()
    repo_root = Path(__file__).resolve().parents[1]
    runtime_root = repo_root / ".runtime"
    tool_root = runtime_root / "tools" / "ffmpeg-n8.1.2-34-g9b6c8969e0" / "bin"
    ffmpeg_path = (tool_root / "ffmpeg").resolve(strict=True)
    ffprobe_path = (tool_root / "ffprobe").resolve(strict=True)
    input_video = arguments.input.resolve(strict=True)
    if not input_video.is_file() or input_video.suffix.casefold() != ".mp4":
        raise ValueError("--input must be an existing MP4 file")

    smoke_root = Path(tempfile.mkdtemp(prefix="flashvsr-real-smoke-", dir=runtime_root))
    chunk_path = smoke_root / "input.npy"
    output_path = smoke_root / "output.npy"
    width, height = _probe_dimensions(ffprobe_path, input_video)
    frames = _decode_first_chunk(
        ffmpeg_path,
        input_video,
        width=width,
        height=height,
    )
    np.save(chunk_path, frames, allow_pickle=False)

    request = WorkerRequest.create(
        job_id="flashvsr-real-resolution-1",
        backend_id="flashvsr-v1.1",
        input_path=chunk_path,
        output_path=output_path,
        parameters={
            "native_scale": 4,
            "output_scale": arguments.output_scale,
            "seed": 0,
            "sparse_ratio": 2.0,
            "kv_ratio": 3.0,
            "local_range": 11,
            "color_fix": True,
            "gpu_index": 0,
        },
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root / "src")
    result = subprocess.run(
        (
            sys.executable,
            str(repo_root / "workers" / "flashvsr_worker.py"),
            "--source-root",
            str(runtime_root / "sources" / "flashvsr-v1.1"),
            "--checkpoint-root",
            str(runtime_root / "checkpoints" / "flashvsr-v1.1"),
        ),
        input=request.to_json(),
        text=True,
        capture_output=True,
        env=environment,
        timeout=3_600,
        check=False,
    )
    if not result.stdout.strip():
        raise RuntimeError(
            f"FlashVSR worker returned no response; stderr={result.stderr.strip()!r}"
        )
    response = WorkerResponse.from_json(result.stdout.strip())
    if result.returncode != 0 or response.status is not WorkerStatus.SUCCEEDED:
        raise RuntimeError(
            f"FlashVSR worker failed: returncode={result.returncode}, "
            f"response={response}, stderr={result.stderr.strip()!r}"
        )
    output = np.load(output_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (
        _FRAME_COUNT,
        height * arguments.output_scale,
        width * arguments.output_scale,
        3,
    )
    if output.shape != expected_shape or output.dtype != np.uint8:
        raise RuntimeError(
            f"FlashVSR output contract mismatch: shape={output.shape}, dtype={output.dtype}, "
            f"expected_shape={expected_shape}"
        )
    print(
        "FlashVSR real-resolution smoke passed:",
        f"input={input_video}",
        f"shape={output.shape}",
        f"dtype={output.dtype}",
        f"sha256={response.output_sha256}",
        f"artifacts={smoke_root}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
