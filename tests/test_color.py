from __future__ import annotations

import unittest

from rvfi_sr.color import RgbDecodeContract, VideoColorRange, VideoColorSpace
from rvfi_sr.probe import ColorMetadata


def metadata(**overrides: str | None) -> ColorMetadata:
    payload: dict[str, str | None] = {
        "pixel_format": "yuv420p",
        "range": None,
        "space": None,
        "transfer": None,
        "primaries": None,
    }
    payload.update(overrides)
    return ColorMetadata(**payload)  # type: ignore[arg-type]


class RgbDecodeContractTest(unittest.TestCase):
    def test_untagged_input_uses_explicit_bt709_limited_assumption(self) -> None:
        contract = RgbDecodeContract.create(
            metadata(),
            untagged_range=VideoColorRange.TV,
            untagged_space=VideoColorSpace.BT709,
        )
        self.assertTrue(contract.assumed_range)
        self.assertTrue(contract.assumed_space)
        self.assertEqual(
            contract.ffmpeg_filter,
            "scale=in_range=tv:in_color_matrix=bt709:out_range=pc,"
            "format=pix_fmts=rgb24",
        )

    def test_explicit_supported_tags_are_not_marked_as_assumptions(self) -> None:
        contract = RgbDecodeContract.create(
            metadata(range="pc", space="bt709"),
            untagged_range=VideoColorRange.TV,
            untagged_space=VideoColorSpace.BT709,
        )
        self.assertFalse(contract.assumed_range)
        self.assertFalse(contract.assumed_space)
        self.assertEqual(contract.source_range, VideoColorRange.PC)

    def test_unsupported_pixel_format_or_matrix_fails_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "pixel format"):
            RgbDecodeContract.create(
                metadata(pixel_format="yuv444p"),
                untagged_range=VideoColorRange.TV,
                untagged_space=VideoColorSpace.BT709,
            )
        with self.assertRaisesRegex(ValueError, "color space"):
            RgbDecodeContract.create(
                metadata(space="bt470bg"),
                untagged_range=VideoColorRange.TV,
                untagged_space=VideoColorSpace.BT709,
            )


if __name__ == "__main__":
    unittest.main()
