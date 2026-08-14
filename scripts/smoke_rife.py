"""RIFE worker를 두 번 실행해 shape, terminal hold, byte 결정성을 검증합니다."""

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
    output_path = smoke_root / f"output-{run_index}.npy"
    request = WorkerRequest.create(
        job_id=f"rife-synthetic-{run_index}",
        backend_id="practical-rife-v4.25",
        input_path=input_path,
        output_path=output_path,
        parameters={
            "temporal_multiplier": 2,
            "fp16": True,
            "inference_scale": 1.0,
            "gpu_index": 0,
        },
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root / "src")
    result = subprocess.run(
        (
            sys.executable,
            str(repo_root / "workers" / "practical_rife_worker.py"),
            "--source-root",
            str(repo_root / ".runtime" / "sources" / "practical-rife-v4.25"),
            "--checkpoint-root",
            str(repo_root / ".runtime" / "checkpoints" / "practical-rife-v4.25"),
        ),
        input=request.to_json(),
        text=True,
        capture_output=True,
        env=environment,
        timeout=120,
        check=False,
    )
    if not result.stdout.strip():
        raise RuntimeError(
            f"RIFE worker returned no protocol response; stderr={result.stderr.strip()!r}"
        )
    response = WorkerResponse.from_json(result.stdout.strip())
    if result.returncode != 0 or response.status is not WorkerStatus.SUCCEEDED:
        raise RuntimeError(
            f"RIFE worker failed: returncode={result.returncode}, "
            f"response={response}, stderr={result.stderr.strip()!r}"
        )
    if response.output_sha256 != _sha256(output_path):
        raise RuntimeError("RIFE response digest does not match the output artifact")
    return response, output_path


def main() -> int:
    """검증용 moving-square 두 프레임으로 repeatability smoke를 수행합니다."""

    repo_root = Path(__file__).resolve().parents[1]
    runtime_root = repo_root / ".runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    smoke_root = Path(tempfile.mkdtemp(prefix="rife-smoke-", dir=runtime_root))
    input_path = smoke_root / "input.npy"

    frames = np.zeros((2, 64, 96, 3), dtype=np.uint8)
    frames[0, 20:36, 16:32] = (255, 64, 32)
    frames[1, 20:36, 48:64] = (255, 64, 32)
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
    if first_output.shape != (4, 64, 96, 3) or first_output.dtype != np.uint8:
        raise RuntimeError(
            f"RIFE output contract mismatch: shape={first_output.shape}, dtype={first_output.dtype}"
        )
    if not np.array_equal(first_output[-1], first_output[-2]):
        raise RuntimeError("RIFE terminal hold frame does not equal the last source frame")
    if not np.array_equal(first_output, second_output):
        difference = np.abs(first_output.astype(np.int16) - second_output.astype(np.int16))
        raise RuntimeError(
            "RIFE repeatability failed: "
            f"changed_values={np.count_nonzero(difference)}, max_delta={difference.max()}"
        )
    if first_response.output_sha256 != second_response.output_sha256:
        raise RuntimeError("byte-identical RIFE outputs reported different digests")

    print(
        "RIFE synthetic smoke passed:",
        f"shape={first_output.shape}",
        f"dtype={first_output.dtype}",
        f"sha256={first_response.output_sha256}",
        f"artifacts={smoke_root}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
