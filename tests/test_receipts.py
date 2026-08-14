from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rvfi_sr.receipts import (
    fingerprint_text,
    receipt_path,
    validate_npy_receipt,
    write_npy_receipt,
)


class NpyArtifactReceiptTest(unittest.TestCase):
    def test_atomic_round_trip_validates_producer_digest_and_shape(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            artifact = Path(temporary_directory) / "frames.npy"
            np.save(artifact, np.zeros((2, 3, 4, 3), dtype=np.uint8))
            producer = fingerprint_text("canonical producer")
            sidecar = write_npy_receipt(artifact, producer_sha256=producer)
            self.assertEqual(sidecar, receipt_path(artifact))
            receipt = validate_npy_receipt(
                artifact,
                producer_sha256=producer,
                expected_shape=(2, 3, 4, 3),
            )
            self.assertEqual(receipt.frame_count, 2)
            self.assertFalse(tuple(artifact.parent.glob("*.partial.json")))
            with self.assertRaises(FileExistsError):
                write_npy_receipt(artifact, producer_sha256=producer)

    def test_changed_producer_or_artifact_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            artifact = Path(temporary_directory) / "frames.npy"
            np.save(artifact, np.zeros((1, 2, 2, 3), dtype=np.uint8))
            producer = fingerprint_text("producer-a")
            write_npy_receipt(artifact, producer_sha256=producer)
            with self.assertRaisesRegex(RuntimeError, "producer mismatch"):
                validate_npy_receipt(
                    artifact,
                    producer_sha256=fingerprint_text("producer-b"),
                    expected_shape=(1, 2, 2, 3),
                )
            with artifact.open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                validate_npy_receipt(
                    artifact,
                    producer_sha256=producer,
                    expected_shape=(1, 2, 2, 3),
                )

    def test_missing_sidecar_and_non_uint8_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            artifact = Path(temporary_directory) / "frames.npy"
            np.save(artifact, np.zeros((1, 2, 2, 3), dtype=np.float32))
            with self.assertRaisesRegex(TypeError, "uint8"):
                write_npy_receipt(
                    artifact,
                    producer_sha256=fingerprint_text("producer"),
                )
            with self.assertRaises(FileNotFoundError):
                validate_npy_receipt(
                    artifact,
                    producer_sha256=fingerprint_text("producer"),
                    expected_shape=(1, 2, 2, 3),
                )
