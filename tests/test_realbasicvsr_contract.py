from __future__ import annotations

import unittest

from rvfi_sr.realbasicvsr_contract import (
    RealBasicVSRResolutionContract,
    plan_realbasicvsr_chunks,
)


class RealBasicVSRResolutionContractTest(unittest.TestCase):
    def test_measured_1920x1072_pair_is_exact_upper_bound(self) -> None:
        contract = RealBasicVSRResolutionContract.create(
            width=1920,
            height=1072,
            model_frames=2,
            output_scale=2,
        )
        self.assertEqual((contract.padded_width, contract.padded_height), (1920, 1088))
        self.assertEqual(contract.padded_frame_pixels, 4_177_920)
        self.assertEqual(
            RealBasicVSRResolutionContract.max_model_frames(width=1920, height=1072),
            2,
        )
        with self.assertRaisesRegex(ValueError, "RTX 3090"):
            RealBasicVSRResolutionContract.create(
                width=1920,
                height=1072,
                model_frames=3,
                output_scale=2,
            )

    def test_604x1080_allows_six_frames_and_rejects_unmeasured_scale(self) -> None:
        self.assertEqual(
            RealBasicVSRResolutionContract.max_model_frames(width=604, height=1080),
            6,
        )
        with self.assertRaisesRegex(ValueError, "output_scale=2"):
            RealBasicVSRResolutionContract.create(
                width=604,
                height=1080,
                model_frames=3,
                output_scale=4,
            )


class RealBasicVSRChunkPlannerTest(unittest.TestCase):
    def test_pair_mode_uses_future_context_and_exact_coverage(self) -> None:
        chunks = plan_realbasicvsr_chunks(
            frame_count=5,
            cut_after=(),
            max_source_frames=2,
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
                (0, 2, 0, 1, 0, 1),
                (1, 3, 0, 1, 1, 2),
                (2, 4, 0, 1, 2, 3),
                (3, 5, 0, 2, 3, 5),
            ],
        )

    def test_larger_chunks_have_bidirectional_context(self) -> None:
        chunks = plan_realbasicvsr_chunks(
            frame_count=10,
            cut_after=(),
            max_source_frames=4,
        )
        self.assertEqual(
            [
                (
                    chunk.source_start,
                    chunk.source_stop,
                    chunk.keep_start,
                    chunk.keep_stop,
                )
                for chunk in chunks
            ],
            [(0, 4, 0, 3), (2, 6, 1, 3), (4, 8, 1, 3), (6, 10, 1, 4)],
        )
        owned = [
            index
            for chunk in chunks
            for index in range(chunk.output_start, chunk.output_stop)
        ]
        self.assertEqual(owned, list(range(10)))

    def test_scene_boundaries_and_one_frame_padding_are_preserved(self) -> None:
        chunks = plan_realbasicvsr_chunks(
            frame_count=6,
            cut_after=(0, 3),
            max_source_frames=4,
        )
        self.assertEqual(chunks[0].pad_terminal, 1)
        self.assertEqual(chunks[0].model_frames, 2)
        self.assertTrue(
            all(
                chunk.source_stop <= chunk.segment_stop
                for chunk in chunks
            )
        )
        owned = [
            index
            for chunk in chunks
            for index in range(chunk.output_start, chunk.output_stop)
        ]
        self.assertEqual(owned, list(range(6)))

    def test_invalid_cut_and_chunk_size_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_source_frames"):
            plan_realbasicvsr_chunks(frame_count=5, cut_after=(), max_source_frames=1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            plan_realbasicvsr_chunks(
                frame_count=5,
                cut_after=(1, 1),
                max_source_frames=2,
            )
