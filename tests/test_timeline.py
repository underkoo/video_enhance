from __future__ import annotations

import unittest
from fractions import Fraction

from rvfi_sr.timeline import SceneCutPolicy, TimelineContract, TransitionKind


class TimelineContractTest(unittest.TestCase):
    def test_multiplier_preserves_duration_and_expands_frame_count(self) -> None:
        contract = TimelineContract.create(
            input_frames=10,
            input_fps=Fraction(30, 1),
            multiplier=2,
            cut_after=(),
            scene_cut_policy=SceneCutPolicy.HOLD_PREVIOUS,
        )
        self.assertEqual(contract.output_frames, 20)
        self.assertEqual(contract.output_fps, Fraction(60, 1))
        self.assertEqual(contract.input_duration, contract.output_duration)
        self.assertEqual(contract.terminal_hold_frames, 1)

    def test_scene_cut_uses_hold_slots_without_shortening_timeline(self) -> None:
        contract = TimelineContract.create(
            input_frames=5,
            input_fps=Fraction(30_000, 1_001),
            multiplier=4,
            cut_after=(1,),
            scene_cut_policy=SceneCutPolicy.HOLD_PREVIOUS,
        )
        transitions = contract.transitions()
        self.assertEqual(len(transitions), 4)
        self.assertEqual(transitions[1].kind, TransitionKind.HOLD_PREVIOUS)
        self.assertEqual(transitions[1].generated_frames, 3)
        self.assertEqual(contract.output_frames, 20)
        self.assertEqual(contract.output_fps, Fraction(120_000, 1_001))
        self.assertEqual(contract.input_duration, contract.output_duration)
        self.assertEqual(contract.terminal_hold_frames, 3)
        self.assertEqual(contract.output_cut_after, (7,))

    def test_duplicate_scene_cut_indices_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate scene-cut"):
            TimelineContract.create(5, Fraction(30, 1), 2, (1, 1))

    def test_out_of_range_scene_cut_fails_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "cut_after"):
            TimelineContract.create(5, Fraction(30, 1), 2, (4,))

    def test_invalid_fps_and_multiplier_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "input_fps"):
            TimelineContract.create(5, Fraction(0, 1), 2, ())
        with self.assertRaisesRegex(ValueError, "multiplier"):
            TimelineContract.create(5, Fraction(30, 1), 1, ())


if __name__ == "__main__":
    unittest.main()
