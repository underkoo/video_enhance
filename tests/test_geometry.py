from __future__ import annotations

import unittest

from rvfi_sr.geometry import AlignedGeometry


class AlignedGeometryTest(unittest.TestCase):
    def test_video_geometry_pads_then_crops_exactly(self) -> None:
        geometry = AlignedGeometry.create(604, 1080, scale=2, alignment=32)
        self.assertEqual((geometry.padded_width, geometry.padded_height), (608, 1088))
        self.assertEqual((geometry.pad_right, geometry.pad_bottom), (4, 8))
        self.assertEqual((geometry.output_width, geometry.output_height), (1208, 2160))
        self.assertEqual(geometry.output_crop_box, (0, 0, 1208, 2160))

    def test_aligned_input_has_zero_padding(self) -> None:
        geometry = AlignedGeometry.create(1280, 960, scale=2, alignment=32)
        self.assertEqual((geometry.pad_right, geometry.pad_bottom), (0, 0))

    def test_non_power_of_two_alignment_fails_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "power of two"):
            AlignedGeometry.create(1280, 720, scale=2, alignment=24)

    def test_invalid_dimensions_and_scale_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "width"):
            AlignedGeometry.create(0, 720, scale=2, alignment=32)
        with self.assertRaisesRegex(ValueError, "scale"):
            AlignedGeometry.create(1280, 720, scale=0, alignment=32)


if __name__ == "__main__":
    unittest.main()
