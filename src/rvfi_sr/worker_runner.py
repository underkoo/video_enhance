"""격리 모델 worker의 terminal JSON과 출력 digest를 검증합니다."""

from __future__ import annotations

import hashlib
import os
import select
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import BinaryIO, Self, cast

from rvfi_sr.worker_protocol import WorkerRequest, WorkerResponse, WorkerStatus

_HASH_CHUNK_BYTES = 8 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_command(command: Sequence[str]) -> tuple[str, ...]:
    if isinstance(command, str) or not command:
        raise ValueError("command must be a non-empty sequence of strings")
    if any(not isinstance(value, str) or not value for value in command):
        raise TypeError("command must contain only non-empty strings")
    # venv의 python은 base interpreter를 가리키는 symlink입니다. 실행 경로를
    # resolve하면 Python이 pyvenv.cfg를 찾지 못해 worker site-packages가 사라집니다.
    executable = Path(command[0]).absolute()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise PermissionError(f"worker executable must be executable: {executable}")
    return (str(executable), *command[1:])


def _validated_timeout(timeout_seconds: int) -> int:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds < 60
    ):
        raise ValueError("timeout_seconds must be an integer >= 60")
    return timeout_seconds


def _merged_environment(environment: Mapping[str, str] | None) -> dict[str, str]:
    if environment is not None and any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        for key, value in environment.items()
    ):
        raise TypeError("environment must map non-empty strings to strings")
    result = os.environ.copy()
    if environment is not None:
        result.update(environment)
    return result


def _validate_terminal_response(
    request: WorkerRequest,
    payload: str,
    *,
    return_code: int | None,
    diagnostic: str,
) -> WorkerResponse:
    if not payload:
        raise RuntimeError(
            f"worker returned no response: returncode={return_code}, stderr={diagnostic}"
        )
    try:
        response = WorkerResponse.from_json(payload)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"worker returned an invalid terminal response: stdout={payload!r}, "
            f"stderr={diagnostic}"
        ) from error
    if response.job_id != request.job_id:
        raise RuntimeError(
            f"worker job_id mismatch: expected={request.job_id!r}, "
            f"actual={response.job_id!r}"
        )
    if (return_code not in {None, 0}) or response.status is not WorkerStatus.SUCCEEDED:
        raise RuntimeError(
            f"worker failed: backend={request.backend_id}, job={request.job_id}, "
            f"returncode={return_code}, error={response.error_type}: "
            f"{response.error_message}, stderr={diagnostic}"
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


def run_worker(
    command: Sequence[str],
    request: WorkerRequest,
    *,
    timeout_seconds: int,
    environment: Mapping[str, str] | None = None,
) -> WorkerResponse:
    """worker 하나를 실행하고 response와 실제 NPY artifact를 교차 검증합니다."""

    validated_command = _validated_command(command)
    if not isinstance(request, WorkerRequest):
        raise TypeError("request must be WorkerRequest")
    validated_timeout = _validated_timeout(timeout_seconds)
    process_environment = _merged_environment(environment)
    try:
        result = subprocess.run(
            validated_command,
            input=request.to_json(),
            capture_output=True,
            text=True,
            check=False,
            timeout=validated_timeout,
            env=process_environment,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"worker timed out: backend={request.backend_id}, job={request.job_id}"
        ) from error
    return _validate_terminal_response(
        request,
        result.stdout.strip(),
        return_code=result.returncode,
        diagnostic=result.stderr.strip(),
    )


class PersistentWorker:
    """모델을 한 번만 적재하는 순차 JSON Lines worker session입니다."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: int,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        validated_command = _validated_command(command)
        self._timeout_seconds = _validated_timeout(timeout_seconds)
        self._error_stream = tempfile.TemporaryFile(mode="w+b")
        self._process = subprocess.Popen(
            validated_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._error_stream,
            env=_merged_environment(environment),
        )
        if self._process.stdin is None or self._process.stdout is None:
            self._process.kill()
            self._process.wait()
            self._error_stream.close()
            raise RuntimeError("persistent worker pipes were not created")
        self._stdin = cast(BinaryIO, self._process.stdin)
        self._stdout = cast(BinaryIO, self._process.stdout)
        self._closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: object,
    ) -> None:
        self.close(check_exit=exception_type is None)

    def _diagnostic(self) -> str:
        self._error_stream.flush()
        self._error_stream.seek(0)
        return self._error_stream.read().decode("utf-8", errors="replace").strip()

    def run(self, request: WorkerRequest) -> WorkerResponse:
        """요청 한 건을 전송하고 대응하는 terminal response와 artifact를 검증합니다."""

        if self._closed:
            raise RuntimeError("persistent worker is already closed")
        if not isinstance(request, WorkerRequest):
            raise TypeError("request must be WorkerRequest")
        if self._process.poll() is not None:
            raise RuntimeError(
                f"persistent worker exited early: returncode={self._process.returncode}, "
                f"stderr={self._diagnostic()}"
            )
        try:
            self._stdin.write(request.to_json().encode("utf-8") + b"\n")
            self._stdin.flush()
        except BrokenPipeError as error:
            raise RuntimeError(
                f"persistent worker input closed early: stderr={self._diagnostic()}"
            ) from error
        readable, _, _ = select.select(
            (self._stdout,),
            (),
            (),
            self._timeout_seconds,
        )
        if not readable:
            self.close(check_exit=False)
            raise RuntimeError(
                f"persistent worker timed out: backend={request.backend_id}, "
                f"job={request.job_id}"
            )
        payload = self._stdout.readline()
        return _validate_terminal_response(
            request,
            payload.decode("utf-8", errors="strict").strip(),
            return_code=self._process.poll(),
            diagnostic=(self._diagnostic() if self._process.poll() is not None else ""),
        )

    def close(self, *, check_exit: bool = True) -> None:
        """stdin EOF로 정상 종료시키고 session process를 회수합니다."""

        if self._closed:
            return
        try:
            self._stdin.close()
        except BrokenPipeError:
            pass
        try:
            return_code = self._process.wait(timeout=self._timeout_seconds)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            return_code = self._process.returncode
        self._stdout.close()
        diagnostic = self._diagnostic()
        self._error_stream.close()
        self._closed = True
        if check_exit and return_code != 0:
            raise RuntimeError(
                f"persistent worker failed on close: returncode={return_code}, "
                f"stderr={diagnostic}"
            )
