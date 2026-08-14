"""격리 모델 worker의 terminal JSON과 출력 digest를 검증합니다."""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from rvfi_sr.worker_protocol import WorkerRequest, WorkerResponse, WorkerStatus

_HASH_CHUNK_BYTES = 8 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def run_worker(
    command: Sequence[str],
    request: WorkerRequest,
    *,
    timeout_seconds: int,
    environment: Mapping[str, str] | None = None,
) -> WorkerResponse:
    """worker 하나를 실행하고 response와 실제 NPY artifact를 교차 검증합니다."""

    if isinstance(command, str) or not command:
        raise ValueError("command must be a non-empty sequence of strings")
    if any(not isinstance(value, str) or not value for value in command):
        raise TypeError("command must contain only non-empty strings")
    # venv의 python은 base interpreter를 가리키는 symlink입니다. 실행 경로를
    # resolve하면 Python이 pyvenv.cfg를 찾지 못해 worker site-packages가 사라집니다.
    executable = Path(command[0]).absolute()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise PermissionError(f"worker executable must be executable: {executable}")
    if not isinstance(request, WorkerRequest):
        raise TypeError("request must be WorkerRequest")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds < 60
    ):
        raise ValueError("timeout_seconds must be an integer >= 60")
    if environment is not None and any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        for key, value in environment.items()
    ):
        raise TypeError("environment must map non-empty strings to strings")

    process_environment = os.environ.copy()
    if environment is not None:
        process_environment.update(environment)
    try:
        result = subprocess.run(
            (str(executable), *command[1:]),
            input=request.to_json(),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
            env=process_environment,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"worker timed out: backend={request.backend_id}, job={request.job_id}"
        ) from error
    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError(
            f"worker returned no response: returncode={result.returncode}, "
            f"stderr={result.stderr.strip()}"
        )
    try:
        response = WorkerResponse.from_json(stdout)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"worker returned an invalid terminal response: stdout={stdout!r}, "
            f"stderr={result.stderr.strip()}"
        ) from error
    if response.job_id != request.job_id:
        raise RuntimeError(
            f"worker job_id mismatch: expected={request.job_id!r}, "
            f"actual={response.job_id!r}"
        )
    if result.returncode != 0 or response.status is not WorkerStatus.SUCCEEDED:
        raise RuntimeError(
            f"worker failed: backend={request.backend_id}, job={request.job_id}, "
            f"returncode={result.returncode}, error={response.error_type}: "
            f"{response.error_message}, stderr={result.stderr.strip()}"
        )
    output_path = Path(request.output_path).resolve(strict=True)
    if not output_path.is_file():
        raise RuntimeError(f"worker output is not a file: {output_path}")
    actual_sha256 = _sha256(output_path)
    if actual_sha256 != response.output_sha256:
        raise RuntimeError(
            f"worker output SHA-256 mismatch: expected={response.output_sha256}, "
            f"actual={actual_sha256}"
        )
    return response
