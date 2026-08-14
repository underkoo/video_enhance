from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rvfi_sr.artifacts import ArtifactContract


class ArtifactContractTest(unittest.TestCase):
    def test_output_is_atomic_and_cannot_alias_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.mp4"
            output_path = root / "output.mp4"
            input_path.touch()
            contract = ArtifactContract.create(input_path, output_path)
            self.assertEqual(contract.partial_path.name, "output.partial.mp4")
            self.assertNotEqual(contract.input_path, contract.output_path)

    def test_same_input_and_output_path_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "video.mp4"
            path.touch()
            with self.assertRaisesRegex(ValueError, "must differ"):
                ArtifactContract.create(path, path)

    def test_missing_input_and_wrong_output_suffix_fail_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaises(FileNotFoundError):
                ArtifactContract.create(root / "missing.mp4", root / "output.mp4")
            input_path = root / "input.mp4"
            input_path.touch()
            with self.assertRaisesRegex(ValueError, r"\.mp4"):
                ArtifactContract.create(input_path, root / "output.avi")


if __name__ == "__main__":
    unittest.main()
