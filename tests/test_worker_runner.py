from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

from rvfi_sr.worker_protocol import WorkerRequest
from rvfi_sr.worker_runner import PersistentWorker, run_worker


class WorkerRunnerTest(unittest.TestCase):
    def _request(self, root: Path, *, job_id: str = "job-1") -> WorkerRequest:
        input_path = root / "input.npy"
        output_path = root / f"{job_id}-output.npy"
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

    def test_persistent_worker_handles_multiple_requests_in_one_process(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            root = Path(temporary_directory)
            first = self._request(root, job_id="job-1")
            second = self._request(root, job_id="job-2")
            script = """
import hashlib
import json
import pathlib
import sys

for line in sys.stdin:
    request = json.loads(line)
    output = request["job_id"].encode("utf-8")
    pathlib.Path(request["output_path"]).write_bytes(output)
    response = {
        "schema_version": 1,
        "job_id": request["job_id"],
        "status": "succeeded",
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "frame_count": 1,
        "width": 1,
        "height": 1,
        "dtype": "uint8",
        "error_type": None,
        "error_message": None,
    }
    print(json.dumps(response, sort_keys=True, separators=(",", ":")), flush=True)
"""
            with PersistentWorker(
                (sys.executable, "-c", script),
                timeout_seconds=60,
            ) as worker:
                first_response = worker.run(first)
                second_response = worker.run(second)
            self.assertEqual(first_response.job_id, "job-1")
            self.assertEqual(second_response.job_id, "job-2")
            self.assertEqual(Path(first.output_path).read_bytes(), b"job-1")
            self.assertEqual(Path(second.output_path).read_bytes(), b"job-2")

    def test_persistent_worker_rejects_calls_after_close(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            request = self._request(Path(temporary_directory))
            worker = PersistentWorker(
                (sys.executable, "-c", "import sys; list(sys.stdin)"),
                timeout_seconds=60,
            )
            worker.close()
            with self.assertRaisesRegex(RuntimeError, "already closed"):
                worker.run(request)

    def test_persistent_worker_surfaces_failed_response_and_reaps_process(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            request = self._request(Path(temporary_directory))
            script = """
import json
import sys

request = json.loads(sys.stdin.readline())
response = {
    "schema_version": 1,
    "job_id": request["job_id"],
    "status": "failed",
    "output_sha256": None,
    "frame_count": None,
    "width": None,
    "height": None,
    "dtype": None,
    "error_type": "RuntimeError",
    "error_message": "synthetic failure",
}
print(json.dumps(response), flush=True)
raise SystemExit(1)
"""
            worker = PersistentWorker(
                (sys.executable, "-c", script),
                timeout_seconds=60,
            )
            try:
                with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                    worker.run(request)
            finally:
                worker.close(check_exit=False)
