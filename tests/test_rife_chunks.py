from __future__ import annotations

import unittest

from rvfi_sr.rife_chunks import plan_rife_chunks


class RifeChunkPlannerTest(unittest.TestCase):
    def test_overlap_ownership_preserves_every_output_slot(self) -> None:
        chunks = plan_rife_chunks(
            frame_count=10,
            cut_after=(),
            multiplier=2,
            max_source_frames=4,
        )
        self.assertEqual(
            [
                (
                    chunk.source_start,
                    chunk.source_stop,
                    chunk.keep_start,
                    chunk.keep_stop,
                    chunk.output_start,
                    chunk.output_stop,
                )
                for chunk in chunks
            ],
            [
                (0, 4, 0, 7, 0, 7),
                (3, 7, 1, 7, 7, 13),
                (6, 10, 1, 8, 13, 20),
            ],
        )

    def test_scene_terminal_hold_and_one_frame_scene_bypass_model(self) -> None:
        chunks = plan_rife_chunks(
            frame_count=5,
            cut_after=(0, 3),
            multiplier=2,
            max_source_frames=3,
        )
        self.assertFalse(chunks[0].use_model)
        self.assertEqual((chunks[0].output_start, chunks[0].output_stop), (0, 2))
        self.assertFalse(chunks[-1].use_model)
        self.assertEqual((chunks[-1].output_start, chunks[-1].output_stop), (8, 10))
        self.assertEqual(
            [(chunk.output_start, chunk.output_stop) for chunk in chunks],
            [(0, 2), (2, 8), (8, 10)],
        )

    def test_multiplier_four_and_sorted_cuts_have_exact_coverage(self) -> None:
        chunks = plan_rife_chunks(
            frame_count=12,
            cut_after=(8, 2),
            multiplier=4,
            max_source_frames=3,
        )
        owned = [
            index
            for chunk in chunks
            for index in range(chunk.output_start, chunk.output_stop)
        ]
        self.assertEqual(owned, list(range(48)))
        self.assertTrue(all(chunk.source_frames <= 3 for chunk in chunks))

    def test_invalid_inputs_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_source_frames"):
            plan_rife_chunks(frame_count=5, cut_after=(), multiplier=2, max_source_frames=1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            plan_rife_chunks(frame_count=5, cut_after=(1, 1), multiplier=2)
        with self.assertRaisesRegex(ValueError, "cut_after"):
            plan_rife_chunks(frame_count=5, cut_after=(4,), multiplier=2)


if __name__ == "__main__":
    unittest.main()
