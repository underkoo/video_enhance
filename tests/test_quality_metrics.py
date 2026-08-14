from __future__ import annotations

import unittest

import numpy as np

from rvfi_sr.quality_metrics import OrderQualityAccumulator


class OrderQualityAccumulatorTest(unittest.TestCase):
    def test_linear_midpoint_has_zero_curvature_and_overshoot(self) -> None:
        source = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        source[1] = 100
        output = np.repeat(source, 2, axis=1).repeat(2, axis=2)
        midpoint = np.full((8, 8, 3), 50, dtype=np.uint8)
        terminal = output[1].copy()
        frames = (output[0], midpoint, output[1], terminal)
        accumulator = OrderQualityAccumulator(
            source,
            output_width=8,
            output_height=8,
            cut_after=(),
            sample_stride=1,
        )
        for index, frame in enumerate(frames):
            accumulator.consume(index, frame.tobytes())
        metrics = accumulator.finalize()
        self.assertEqual(metrics.source_roundtrip_psnr_db, 100.0)
        self.assertEqual(metrics.source_roundtrip_mae, 0.0)
        self.assertEqual(metrics.midpoint_curvature_mae, 0.0)
        self.assertEqual(metrics.midpoint_overshoot_rate, 0.0)
        self.assertEqual(metrics.midpoint_frames, 1)

    def test_scene_cut_midpoint_is_excluded(self) -> None:
        source = np.zeros((3, 4, 4, 3), dtype=np.uint8)
        output = np.repeat(source, 2, axis=1).repeat(2, axis=2)
        frames = (output[0], output[0], output[1], output[1], output[2], output[2])
        accumulator = OrderQualityAccumulator(
            source,
            output_width=8,
            output_height=8,
            cut_after=(0,),
            sample_stride=1,
        )
        for index, frame in enumerate(frames):
            accumulator.consume(index, frame.tobytes())
        metrics = accumulator.finalize()
        self.assertEqual(metrics.midpoint_frames, 1)

    def test_nonsequential_and_short_streams_fail_fast(self) -> None:
        source = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        accumulator = OrderQualityAccumulator(
            source,
            output_width=8,
            output_height=8,
            cut_after=(),
        )
        with self.assertRaisesRegex(ValueError, "sequential"):
            accumulator.consume(1, bytes(8 * 8 * 3))
        accumulator.consume(0, bytes(8 * 8 * 3))
        with self.assertRaisesRegex(RuntimeError, "ended early"):
            accumulator.finalize()

    def test_input_without_a_measurable_transition_fails_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            OrderQualityAccumulator(
                np.zeros((1, 4, 4, 3), dtype=np.uint8),
                output_width=8,
                output_height=8,
                cut_after=(),
            )
        with self.assertRaisesRegex(ValueError, "non-cut transition"):
            OrderQualityAccumulator(
                np.zeros((2, 4, 4, 3), dtype=np.uint8),
                output_width=8,
                output_height=8,
                cut_after=(0,),
            )
