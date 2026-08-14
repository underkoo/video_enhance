from __future__ import annotations

import unittest

from rvfi_sr.temporal_chunks import plan_flashvsr_chunks


class FlashVSRTemporalChunkPlanTest(unittest.TestCase):
    def test_long_segment_has_context_overlap_without_output_duplication(self) -> None:
        chunks = plan_flashvsr_chunks(frame_count=50, cut_after=())
        self.assertEqual(
            [
                (
                    chunk.source_start,
                    chunk.source_stop,
                    chunk.pad_terminal,
                    chunk.keep_start,
                    chunk.keep_stop,
                )
                for chunk in chunks
            ],
            [
                (0, 21, 0, 0, 21),
                (16, 37, 0, 5, 21),
                (32, 50, 3, 5, 18),
            ],
        )
        kept_indices = [
            index
            for chunk in chunks
            for index in range(chunk.output_start, chunk.output_stop)
        ]
        self.assertEqual(kept_indices, list(range(50)))

    def test_scene_cut_starts_new_chunk_without_crossing_boundary(self) -> None:
        chunks = plan_flashvsr_chunks(frame_count=30, cut_after=(9,))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(
            (chunks[0].segment_start, chunks[0].segment_stop),
            (0, 10),
        )
        self.assertEqual(
            (chunks[0].source_start, chunks[0].source_stop, chunks[0].pad_terminal),
            (0, 10, 11),
        )
        self.assertEqual(
            (chunks[1].segment_start, chunks[1].segment_stop),
            (10, 30),
        )
        self.assertEqual(
            (chunks[1].source_start, chunks[1].source_stop, chunks[1].pad_terminal),
            (10, 30, 1),
        )

    def test_single_frame_segment_is_terminal_padded_but_kept_once(self) -> None:
        chunks = plan_flashvsr_chunks(frame_count=3, cut_after=(0, 1))
        self.assertEqual(len(chunks), 3)
        for expected_index, chunk in enumerate(chunks):
            self.assertEqual(chunk.source_start, expected_index)
            self.assertEqual(chunk.source_stop, expected_index + 1)
            self.assertEqual(chunk.pad_terminal, 20)
            self.assertEqual(
                (chunk.output_start, chunk.output_stop),
                (expected_index, expected_index + 1),
            )

    def test_invalid_cut_and_frame_count_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            plan_flashvsr_chunks(frame_count=0, cut_after=())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            plan_flashvsr_chunks(frame_count=10, cut_after=(4, 4))
        with self.assertRaisesRegex(ValueError, "cut_after"):
            plan_flashvsr_chunks(frame_count=10, cut_after=(9,))


if __name__ == "__main__":
    unittest.main()
