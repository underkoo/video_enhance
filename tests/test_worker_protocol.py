from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rvfi_sr.worker_protocol import WorkerRequest, WorkerResponse, WorkerStatus


class WorkerProtocolTest(unittest.TestCase):
    def test_request_round_trip_preserves_absolute_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            request = WorkerRequest.create(
                job_id="smoke-0001",
                backend_id="practical-rife-v4.25",
                input_path=root / "frames-input.npz",
                output_path=root / "frames-output.npz",
                parameters={"temporal_multiplier": 2, "fp16": True},
            )
            restored = WorkerRequest.from_json(request.to_json())
            self.assertEqual(restored, request)

    def test_relative_and_aliased_paths_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "absolute"):
            WorkerRequest.create(
                job_id="smoke-0001",
                backend_id="practical-rife-v4.25",
                input_path=Path("input.npz"),
                output_path=Path("output.npz"),
                parameters={},
            )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory).resolve() / "same.npz"
            with self.assertRaisesRegex(ValueError, "must differ"):
                WorkerRequest.create(
                    job_id="smoke-0001",
                    backend_id="practical-rife-v4.25",
                    input_path=path,
                    output_path=path,
                    parameters={},
                )

    def test_unknown_schema_and_parameters_fail_fast(self) -> None:
        payload = {
            "schema_version": 99,
            "job_id": "smoke-0001",
            "backend_id": "practical-rife-v4.25",
            "input_path": "/tmp/input.npz",
            "output_path": "/tmp/output.npz",
            "parameters": {},
        }
        with self.assertRaisesRegex(ValueError, "schema_version"):
            WorkerRequest.from_json(json.dumps(payload))

        payload["schema_version"] = 1
        payload["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "unexpected"):
            WorkerRequest.from_json(json.dumps(payload))

    def test_success_response_requires_output_digest_and_shape(self) -> None:
        response = WorkerResponse(
            schema_version=1,
            job_id="smoke-0001",
            status=WorkerStatus.SUCCEEDED,
            output_sha256="b" * 64,
            frame_count=20,
            width=1280,
            height=720,
            dtype="uint8",
        )
        self.assertEqual(WorkerResponse.from_json(response.to_json()), response)
        with self.assertRaisesRegex(ValueError, "output_sha256"):
            WorkerResponse(
                schema_version=1,
                job_id="smoke-0001",
                status=WorkerStatus.SUCCEEDED,
                output_sha256=None,
                frame_count=20,
                width=1280,
                height=720,
                dtype="uint8",
            )


if __name__ == "__main__":
    unittest.main()
