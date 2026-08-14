"""RealBasicVSR의 독립/persistent strict load와 byte repeatability를 검사합니다."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from rvfi_sr.worker_protocol import WorkerRequest, WorkerResponse, WorkerStatus
from rvfi_sr.worker_runner import PersistentWorker


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _run_worker(
    repo_root: Path,
    checkpoint_path: Path,
    input_path: Path,
    output_path: Path,
    run_index: int,
) -> WorkerResponse:
    request = WorkerRequest.create(
        job_id=f"realbasicvsr-synthetic-{run_index}",
        backend_id="mmagic-realbasicvsr",
        input_path=input_path,
        output_path=output_path,
        parameters={
            "native_scale": 4,
            "output_scale": 2,
            "fp16": True,
            "gpu_index": 0,
        },
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root / "src")
    result = subprocess.run(
        (
            sys.executable,
            str(repo_root / "workers" / "realbasicvsr_worker.py"),
            "--checkpoint",
            str(checkpoint_path),
        ),
        input=request.to_json(),
        text=True,
        capture_output=True,
        env=environment,
        timeout=300,
        check=False,
    )
    if not result.stdout.strip():
        raise RuntimeError(
            "RealBasicVSR worker returned no response: "
            f"returncode={result.returncode}, stderr={result.stderr[-4000:]!r}"
        )
    response = WorkerResponse.from_json(result.stdout.strip())
    if result.returncode != 0 or response.status is not WorkerStatus.SUCCEEDED:
        raise RuntimeError(
            f"RealBasicVSR worker failed: returncode={result.returncode}, "
            f"response={response}, stderr={result.stderr[-4000:]!r}"
        )
    if response.output_sha256 != _sha256(output_path):
        raise RuntimeError("RealBasicVSR response digest does not match output")
    return response


def _request(
    *,
    input_path: Path,
    output_path: Path,
    run_index: int,
) -> WorkerRequest:
    return WorkerRequest.create(
        job_id=f"realbasicvsr-persistent-{run_index}",
        backend_id="mmagic-realbasicvsr",
        input_path=input_path,
        output_path=output_path,
        parameters={
            "native_scale": 4,
            "output_scale": 2,
            "fp16": True,
            "gpu_index": 0,
        },
    )


def main() -> int:
    """64x96 입력을 독립 2회와 동일 persistent session 2회로 처리합니다."""

    repo_root = Path(__file__).resolve().parents[1]
    checkpoint_path = (
        repo_root / ".runtime" / "checkpoints" / "mmagic-realbasicvsr" / "RealBasicVSR.pth"
    )
    with tempfile.TemporaryDirectory(
        prefix="realbasicvsr-smoke-",
        dir=repo_root / ".runtime",
    ) as temporary_directory:
        smoke_root = Path(temporary_directory)
        input_path = smoke_root / "input.npy"
        frames = np.zeros((3, 64, 96, 3), dtype=np.uint8)
        frames[0, 16:40, 8:32] = (255, 48, 16)
        frames[1, 16:40, 32:56] = (255, 48, 16)
        frames[2, 16:40, 56:80] = (255, 48, 16)
        np.save(input_path, frames, allow_pickle=False)

        responses: list[WorkerResponse] = []
        outputs: list[np.ndarray] = []
        for run_index in (1, 2):
            output_path = smoke_root / f"output-{run_index}.npy"
            responses.append(
                _run_worker(
                    repo_root,
                    checkpoint_path,
                    input_path,
                    output_path,
                    run_index,
                )
            )
            outputs.append(np.load(output_path, allow_pickle=False))
        environment = {"PYTHONPATH": str(repo_root / "src")}
        with PersistentWorker(
            (
                sys.executable,
                str(repo_root / "workers" / "realbasicvsr_worker.py"),
                "--checkpoint",
                str(checkpoint_path),
                "--persistent",
            ),
            timeout_seconds=300,
            environment=environment,
        ) as persistent_worker:
            for run_index in (3, 4):
                output_path = smoke_root / f"output-{run_index}.npy"
                responses.append(
                    persistent_worker.run(
                        _request(
                            input_path=input_path,
                            output_path=output_path,
                            run_index=run_index,
                        )
                    )
                )
                outputs.append(np.load(output_path, allow_pickle=False))

        if outputs[0].shape != (3, 128, 192, 3) or outputs[0].dtype != np.uint8:
            raise RuntimeError(
                f"RealBasicVSR output contract mismatch: "
                f"shape={outputs[0].shape}, dtype={outputs[0].dtype}"
            )
        for output in outputs[1:]:
            if not np.array_equal(outputs[0], output):
                difference = np.abs(
                    outputs[0].astype(np.int16) - output.astype(np.int16)
                )
                raise RuntimeError(
                    "RealBasicVSR repeatability failed: "
                    f"changed_values={np.count_nonzero(difference)}, "
                    f"max_delta={difference.max()}"
                )
        if len({response.output_sha256 for response in responses}) != 1:
            raise RuntimeError("byte-identical outputs reported different digests")
        print(
            "RealBasicVSR synthetic smoke passed:",
            f"shape={outputs[0].shape}",
            f"sha256={responses[0].output_sha256}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
