from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rvfi_sr.provenance import (
    CheckpointArtifact,
    ModelProvenance,
    write_provenance_manifest,
)


def make_provenance() -> ModelProvenance:
    return ModelProvenance(
        backend_id="practical-rife-v4.25",
        upstream_url="https://github.com/hzwer/Practical-RIFE",
        upstream_commit="17d8c7a1005b37f4c97bfee04e316aaec7fdc536",
        license_name="MIT",
        checkpoints=(
            CheckpointArtifact(
                filename="RIFEv4.25.zip",
                url="https://drive.usercontent.google.com/download?id=immutable",
                sha256="a" * 64,
                size_bytes=22_919_050,
            ),
        ),
    )


class ModelProvenanceTest(unittest.TestCase):
    def test_requires_immutable_commit_and_checkpoint_digest(self) -> None:
        with self.assertRaisesRegex(ValueError, "upstream_commit"):
            replace(make_provenance(), upstream_commit="main")
        with self.assertRaisesRegex(ValueError, "sha256"):
            CheckpointArtifact(
                filename="model.ckpt",
                url="https://example.invalid/model.ckpt",
                sha256="unknown",
                size_bytes=1,
            )

    def test_requires_https_official_locations(self) -> None:
        with self.assertRaisesRegex(ValueError, "upstream_url"):
            replace(
                make_provenance(),
                upstream_url="git@github.com:hzwer/Practical-RIFE",
            )
        with self.assertRaisesRegex(ValueError, "url"):
            CheckpointArtifact(
                filename="model.ckpt",
                url="http://example.invalid/model.ckpt",
                sha256="a" * 64,
                size_bytes=1,
            )

    def test_checkpoint_filename_and_size_are_safe(self) -> None:
        with self.assertRaisesRegex(ValueError, "filename"):
            CheckpointArtifact(
                filename="../model.ckpt",
                url="https://example.invalid/model.ckpt",
                sha256="a" * 64,
                size_bytes=1,
            )
        with self.assertRaisesRegex(ValueError, "size_bytes"):
            CheckpointArtifact(
                filename="model.ckpt",
                url="https://example.invalid/model.ckpt",
                sha256="a" * 64,
                size_bytes=0,
            )

    def test_manifest_write_is_canonical_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "provenance.json"
            write_provenance_manifest(output_path, (make_provenance(),))
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["models"][0]["backend_id"], "practical-rife-v4.25")
            self.assertEqual(
                payload["models"][0]["checkpoints"][0]["size_bytes"],
                22_919_050,
            )
            self.assertTrue(output_path.read_bytes().endswith(b"\n"))
            with self.assertRaises(FileExistsError):
                write_provenance_manifest(output_path, (make_provenance(),))
            self.assertFalse((output_path.parent / "provenance.partial.json").exists())


if __name__ == "__main__":
    unittest.main()
