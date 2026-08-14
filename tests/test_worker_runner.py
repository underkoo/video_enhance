from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

from rvfi_sr.worker_protocol import WorkerRequest
from rvfi_sr.worker_runner import run_worker


class WorkerRunnerTest(unittest.TestCase):
    def _request(self, root: Path, *, job_id: str = "job-1") -> WorkerRequest:
        input_path = root / "input.npy"
        output_path = root / "output.npy"
        input_path.write_bytes(b"input")
        return WorkerRequest.create(
            job_id=job_id,
            backend_id="fake-backend",
            input_path=input_path,
            output_path=output_path,
            parameters={},
        )

    def test_validates_response_and_actual_output_digest(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            root = Path(temporary_directory)
            request = self._request(root)
            output = b"verified output"
            digest = hashlib.sha256(output).hexdigest()
            script = (
                "import pathlib,sys; "
                f"pathlib.Path({request.output_path!r}).write_bytes({output!r}); "
                "sys.stdin.read(); "
                "print('{\"dtype\":\"uint8\",\"error_message\":null,"
                "\"error_type\":null,\"frame_count\":1,\"height\":1,"
                f"\"job_id\":\"job-1\",\"output_sha256\":\"{digest}\","
                "\"schema_version\":1,\"status\":\"succeeded\",\"width\":1}')"
            )
            response = run_worker(
                (sys.executable, "-c", script),
                request,
                timeout_seconds=60,
            )
            self.assertEqual(response.output_sha256, digest)

    def test_rejects_digest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            root = Path(temporary_directory)
            request = self._request(root)
            script = (
                "import pathlib,sys; "
                f"pathlib.Path({request.output_path!r}).write_bytes(b'corrupt'); "
                "sys.stdin.read(); "
                "print('{\"dtype\":\"uint8\",\"error_message\":null,"
                "\"error_type\":null,\"frame_count\":1,\"height\":1,"
                "\"job_id\":\"job-1\",\"output_sha256\":\""
                + "a" * 64
                + "\",\"schema_version\":1,\"status\":\"succeeded\","
                "\"width\":1}')"
            )
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                run_worker(
                    (sys.executable, "-c", script),
                    request,
                    timeout_seconds=60,
                )

    def test_failed_worker_surfaces_terminal_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            request = self._request(Path(temporary_directory))
            script = (
                "import sys; sys.stdin.read(); "
                "print('{\"dtype\":null,\"error_message\":\"OOM\","
                "\"error_type\":\"RuntimeError\",\"frame_count\":null,"
                "\"height\":null,\"job_id\":\"job-1\","
                "\"output_sha256\":null,\"schema_version\":1,"
                "\"status\":\"failed\",\"width\":null}'); sys.exit(1)"
            )
            with self.assertRaisesRegex(RuntimeError, "RuntimeError: OOM"):
                run_worker(
                    (sys.executable, "-c", script),
                    request,
                    timeout_seconds=60,
                )
