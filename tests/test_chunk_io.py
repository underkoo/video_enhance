from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rvfi_sr.chunk_io import (
    FlashVSRInputAssembler,
    RifeInputAssembler,
    stream_rife_output_frames,
)
from rvfi_sr.rife_chunks import plan_rife_chunks
from rvfi_sr.temporal_chunks import plan_flashvsr_chunks


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

    def test_rife_output_ownership_removes_chunk_terminal_holds(self) -> None:
        chunks = plan_rife_chunks(
            frame_count=5,
            cut_after=(),
            multiplier=2,
            max_source_frames=3,
        )
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            root = Path(temporary_directory)
            input_paths: list[Path] = []
            output_paths: list[Path] = []
            for chunk_index, chunk in enumerate(chunks):
                input_path = root / f"input-{chunk_index}.npy"
                output_path = root / f"output-{chunk_index}.npy"
                np.save(
                    input_path,
                    np.zeros((chunk.source_frames, 1, 1, 3), dtype=np.uint8),
                    allow_pickle=False,
                )
                values = np.arange(
                    chunk.output_start - chunk.keep_start,
                    chunk.output_start - chunk.keep_start + chunk.worker_output_frames,
                    dtype=np.uint8,
                )
                np.save(
                    output_path,
                    np.repeat(values[:, None, None, None], 3, axis=3),
                    allow_pickle=False,
                )
                input_paths.append(input_path)
                output_paths.append(output_path)
            received: list[int] = []
            stream_rife_output_frames(
                chunks,
                tuple(input_paths),
                tuple(output_paths),
                width=1,
                height=1,
                consume_frame=lambda _index, frame: received.append(frame[0]),
            )
            self.assertEqual(received, list(range(10)))

    def test_rife_output_single_frame_scene_repeats_without_worker(self) -> None:
        chunks = plan_rife_chunks(frame_count=1, cut_after=(), multiplier=2)
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            input_path = Path(temporary_directory) / "input.npy"
            np.save(input_path, np.full((1, 1, 1, 3), 17, dtype=np.uint8))
            received: list[bytes] = []
            stream_rife_output_frames(
                chunks,
                (input_path,),
                (None,),
                width=1,
                height=1,
                consume_frame=lambda _index, frame: received.append(frame),
            )
            self.assertEqual(received, [bytes([17, 17, 17])] * 2)


class FlashVSRInputAssemblerTest(unittest.TestCase):
    def test_context_overlap_and_terminal_padding_are_exact(self) -> None:
        chunks = plan_flashvsr_chunks(frame_count=30, cut_after=())
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            assembler = FlashVSRInputAssembler(
                chunks,
                width=1,
                height=1,
                output_dir=Path(temporary_directory) / "flash",
            )
            for frame_index in range(30):
                assembler.consume(frame_index, bytes([frame_index] * 3))
            paths = assembler.finalize()
            self.assertEqual(len(paths), 2)
            first = np.load(paths[0], allow_pickle=False)[:, 0, 0, 0]
            second = np.load(paths[1], allow_pickle=False)[:, 0, 0, 0]
            self.assertTrue(np.array_equal(first, np.arange(21)))
            self.assertTrue(np.array_equal(second[:14], np.arange(16, 30)))
            self.assertTrue(np.all(second[14:] == 29))

    def test_scene_chunks_never_share_context_across_cut(self) -> None:
        chunks = plan_flashvsr_chunks(frame_count=8, cut_after=(2,))
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            assembler = FlashVSRInputAssembler(
                chunks,
                width=1,
                height=1,
                output_dir=Path(temporary_directory) / "flash",
            )
            for frame_index in range(8):
                assembler.consume(frame_index, bytes([frame_index] * 3))
            paths = assembler.finalize()
            left = np.load(paths[0], allow_pickle=False)[:, 0, 0, 0]
            right = np.load(paths[1], allow_pickle=False)[:, 0, 0, 0]
            self.assertTrue(np.array_equal(left[:3], [0, 1, 2]))
            self.assertTrue(np.all(left[3:] == 2))
            self.assertTrue(np.array_equal(right[:5], [3, 4, 5, 6, 7]))
            self.assertTrue(np.all(right[5:] == 7))


if __name__ == "__main__":
    unittest.main()
