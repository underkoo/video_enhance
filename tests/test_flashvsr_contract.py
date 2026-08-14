from __future__ import annotations

import unittest

from rvfi_sr.flashvsr_contract import (
    FlashVSRChunkContract,
    FlashVSRResolutionContract,
)


class FlashVSRChunkContractTest(unittest.TestCase):
    def test_minimum_chunk_has_valid_internal_frame_count(self) -> None:
        contract = FlashVSRChunkContract.create(source_frames=21)
        self.assertEqual(contract.padded_frames, 25)
        self.assertEqual(contract.process_iterations, 1)

    def test_larger_valid_chunk_preserves_every_source_frame(self) -> None:
        contract = FlashVSRChunkContract.create(source_frames=29)
        self.assertEqual(contract.padded_frames, 33)
        self.assertEqual(contract.process_iterations, 2)

    def test_invalid_or_too_short_chunk_fails_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 21"):
            FlashVSRChunkContract.create(source_frames=13)
        with self.assertRaisesRegex(ValueError, "8n - 3"):
            FlashVSRChunkContract.create(source_frames=22)

    def test_bool_is_not_accepted_as_frame_count(self) -> None:
        with self.assertRaisesRegex(TypeError, "integer"):
            FlashVSRChunkContract.create(source_frames=True)


class FlashVSRResolutionContractTest(unittest.TestCase):
    def test_measured_3090_limit_accepts_tested_portrait_resolution(self) -> None:
        contract = FlashVSRResolutionContract.create(width=604, height=1080)
        self.assertEqual((contract.padded_width, contract.padded_height), (608, 1088))
        self.assertEqual(contract.padded_pixels, 661_504)

    def test_larger_single_pass_input_fails_before_model_load(self) -> None:
        with self.assertRaisesRegex(ValueError, "single-pass limit"):
            FlashVSRResolutionContract.create(width=1280, height=720)

    def test_invalid_dimensions_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "width"):
            FlashVSRResolutionContract.create(width=0, height=1080)


if __name__ == "__main__":
    unittest.main()
