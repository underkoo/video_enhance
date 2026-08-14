"""FlashVSR v1.1 worker의 실제 checkpoint 추론 smoke를 수행합니다."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from rvfi_sr.worker_protocol import WorkerRequest, WorkerResponse, WorkerStatus


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _run_worker(
    *,
    repo_root: Path,
    smoke_root: Path,
    input_path: Path,
    run_index: int,
) -> tuple[WorkerResponse, Path]:
    runtime_root = repo_root / ".runtime"
    output_path = smoke_root / f"output-{run_index}.npy"
    request = WorkerRequest.create(
        job_id=f"flashvsr-synthetic-{run_index}",
        backend_id="flashvsr-v1.1",
        input_path=input_path,
        output_path=output_path,
        parameters={
            "native_scale": 4,
            "output_scale": 2,
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
        timeout=1_800,
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
    if response.output_sha256 != _sha256(output_path):
        raise RuntimeError("FlashVSR response digest does not match the output artifact")
    return response, output_path


def main() -> int:
    """21-frame synthetic motion을 native 4x 후 explicit 2x로 검증합니다."""

    repo_root = Path(__file__).resolve().parents[1]
    runtime_root = repo_root / ".runtime"
    smoke_root = Path(tempfile.mkdtemp(prefix="flashvsr-smoke-", dir=runtime_root))
    input_path = smoke_root / "input.npy"

    frames = np.zeros((21, 32, 32, 3), dtype=np.uint8)
    for frame_index in range(frames.shape[0]):
        left = min(frame_index, 24)
        frames[frame_index, 10:18, left : left + 8] = (240, 96, 32)
    np.save(input_path, frames, allow_pickle=False)

    first_response, first_output_path = _run_worker(
        repo_root=repo_root,
        smoke_root=smoke_root,
        input_path=input_path,
        run_index=1,
    )
    second_response, second_output_path = _run_worker(
        repo_root=repo_root,
        smoke_root=smoke_root,
        input_path=input_path,
        run_index=2,
    )
    first_output = np.load(first_output_path, allow_pickle=False)
    second_output = np.load(second_output_path, allow_pickle=False)
    if first_output.shape != (21, 64, 64, 3) or first_output.dtype != np.uint8:
        raise RuntimeError(
            "FlashVSR output contract mismatch: "
            f"shape={first_output.shape}, dtype={first_output.dtype}"
        )
    if not np.array_equal(first_output, second_output):
        difference = np.abs(first_output.astype(np.int16) - second_output.astype(np.int16))
        raise RuntimeError(
            "FlashVSR repeatability failed: "
            f"changed_values={np.count_nonzero(difference)}, max_delta={difference.max()}"
        )
    if first_response.output_sha256 != second_response.output_sha256:
        raise RuntimeError("byte-identical FlashVSR outputs reported different digests")
    print(
        "FlashVSR synthetic smoke passed:",
        f"shape={first_output.shape}",
        f"dtype={first_output.dtype}",
        f"range=[{first_output.min()}, {first_output.max()}]",
        f"sha256={first_response.output_sha256}",
        f"artifacts={smoke_root}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
