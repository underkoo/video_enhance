from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from rvfi_sr.model_specs import MODEL_PROVENANCE
from rvfi_sr.provenance import CheckpointArtifact, verify_checkpoint
from rvfi_sr.registry import BACKEND_REGISTRY


class ModelSpecsTest(unittest.TestCase):
    def test_default_model_specs_match_registered_backends(self) -> None:
        for backend_id, provenance in MODEL_PROVENANCE.items():
            self.assertIn(backend_id, BACKEND_REGISTRY)
            self.assertEqual(backend_id, provenance.backend_id)

    def test_flashvsr_declares_all_four_weight_files(self) -> None:
        checkpoints = MODEL_PROVENANCE["flashvsr-v1.1"].checkpoints
        self.assertEqual(
            {checkpoint.filename for checkpoint in checkpoints},
            {
                "diffusion_pytorch_model_streaming_dmd.safetensors",
                "LQ_proj_in.ckpt",
                "TCDecoder.ckpt",
                "Wan2.1_VAE.pth",
            },
        )
        self.assertEqual(sum(item.size_bytes for item in checkpoints), 6_948_393_553)

    def test_checkpoint_verifier_checks_size_then_digest(self) -> None:
        content = b"verified checkpoint bytes"
        spec = CheckpointArtifact(
            filename="model.ckpt",
            url="https://example.invalid/model.ckpt",
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
        )
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "model.ckpt"
            checkpoint_path.write_bytes(content)
            verify_checkpoint(checkpoint_path, spec)
            checkpoint_path.write_bytes(content + b"corrupt")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                verify_checkpoint(checkpoint_path, spec)

            wrong_digest_spec = CheckpointArtifact(
                filename="model.ckpt",
                url=spec.url,
                sha256="a" * 64,
                size_bytes=checkpoint_path.stat().st_size,
            )
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_checkpoint(checkpoint_path, wrong_digest_spec)


if __name__ == "__main__":
    unittest.main()
