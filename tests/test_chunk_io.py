from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rvfi_sr.chunk_io import RifeInputAssembler
from rvfi_sr.rife_chunks import plan_rife_chunks


class RifeInputAssemblerTest(unittest.TestCase):
    def test_streaming_assembly_writes_atomic_overlapping_chunks(self) -> None:
        chunks = plan_rife_chunks(
            frame_count=5,
            cut_after=(),
            multiplier=2,
            max_source_frames=3,
        )
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            output_dir = Path(temporary_directory) / "chunks"
            assembler = RifeInputAssembler(
                chunks,
                width=2,
                height=1,
                output_dir=output_dir,
            )
            for frame_index in range(5):
                assembler.consume(frame_index, bytes([frame_index] * 6))
            paths = assembler.finalize()
            self.assertEqual(len(paths), 2)
            first = np.load(paths[0], allow_pickle=False)
            second = np.load(paths[1], allow_pickle=False)
            self.assertEqual(first.shape, (3, 1, 2, 3))
            self.assertEqual(second.shape, (3, 1, 2, 3))
            self.assertTrue(np.all(first[:, 0, 0, 0] == [0, 1, 2]))
            self.assertTrue(np.all(second[:, 0, 0, 0] == [2, 3, 4]))
            self.assertFalse(tuple(output_dir.glob("*.partial.npy")))

    def test_scene_boundary_does_not_share_source_frames(self) -> None:
        chunks = plan_rife_chunks(
            frame_count=4,
            cut_after=(1,),
            multiplier=2,
            max_source_frames=4,
        )
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            assembler = RifeInputAssembler(
                chunks,
                width=1,
                height=1,
                output_dir=Path(temporary_directory) / "chunks",
            )
            for frame_index in range(4):
                assembler.consume(frame_index, bytes([frame_index] * 3))
            paths = assembler.finalize()
            self.assertEqual(np.load(paths[0], allow_pickle=False).shape[0], 2)
            self.assertEqual(np.load(paths[1], allow_pickle=False).shape[0], 2)

    def test_nonsequential_short_and_incomplete_streams_fail_fast(self) -> None:
        chunks = plan_rife_chunks(frame_count=2, cut_after=(), multiplier=2)
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            assembler = RifeInputAssembler(
                chunks,
                width=2,
                height=1,
                output_dir=Path(temporary_directory) / "chunks",
            )
            with self.assertRaisesRegex(ValueError, "sequential"):
                assembler.consume(1, bytes(6))
            with self.assertRaisesRegex(ValueError, "byte count"):
                assembler.consume(0, bytes(5))
            assembler.consume(0, bytes(6))
            with self.assertRaisesRegex(RuntimeError, "ended early"):
                assembler.finalize()


if __name__ == "__main__":
    unittest.main()
