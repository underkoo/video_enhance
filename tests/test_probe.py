from __future__ import annotations

import unittest
from fractions import Fraction

from rvfi_sr.probe import parse_ffprobe_payload


def make_payload() -> dict[str, object]:
    return {
        "format": {"duration": "10.010000", "format_name": "mov,mp4"},
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 604,
                "height": 1080,
                "pix_fmt": "yuv420p",
                "avg_frame_rate": "30000/1001",
                "r_frame_rate": "30000/1001",
                "time_base": "1/90000",
                "nb_read_frames": "300",
                "color_range": "tv",
                "color_space": "bt709",
                "color_transfer": "bt709",
                "color_primaries": "bt709",
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
                "channels": 2,
            },
        ],
    }


class ProbeParserTest(unittest.TestCase):
    def test_preserves_rational_rate_color_and_audio_metadata(self) -> None:
        spec = parse_ffprobe_payload(make_payload())
        self.assertEqual(spec.width, 604)
        self.assertEqual(spec.height, 1080)
        self.assertEqual(spec.average_fps, Fraction(30_000, 1_001))
        self.assertEqual(spec.nominal_fps, Fraction(30_000, 1_001))
        self.assertEqual(spec.time_base, Fraction(1, 90_000))
        self.assertEqual(spec.frame_count, 300)
        self.assertEqual(spec.duration, Fraction(1001, 100))
        self.assertEqual(spec.color.space, "bt709")
        self.assertEqual(spec.audio.codec, "aac")
        self.assertFalse(spec.requires_cfr_normalization)

    def test_rate_mismatch_requires_cfr_normalization(self) -> None:
        payload = make_payload()
        payload["streams"][0]["r_frame_rate"] = "30/1"
        spec = parse_ffprobe_payload(payload)
        self.assertTrue(spec.requires_cfr_normalization)

    def test_audio_is_optional(self) -> None:
        payload = make_payload()
        payload["streams"] = [payload["streams"][0]]
        spec = parse_ffprobe_payload(payload)
        self.assertIsNone(spec.audio)

    def test_missing_color_tags_are_preserved_as_unknown(self) -> None:
        payload = make_payload()
        video = payload["streams"][0]
        del video["color_range"]
        del video["color_space"]
        del video["color_transfer"]
        del video["color_primaries"]
        spec = parse_ffprobe_payload(payload)
        self.assertIsNone(spec.color.range)
        self.assertIsNone(spec.color.space)
        self.assertIsNone(spec.color.transfer)
        self.assertIsNone(spec.color.primaries)
        self.assertFalse(spec.color.is_complete)

    def test_multiple_video_streams_fail_fast(self) -> None:
        payload = make_payload()
        payload["streams"].append(dict(payload["streams"][0]))
        with self.assertRaisesRegex(ValueError, "exactly one video"):
            parse_ffprobe_payload(payload)

    def test_missing_count_and_invalid_rate_fail_fast(self) -> None:
        payload = make_payload()
        del payload["streams"][0]["nb_read_frames"]
        with self.assertRaisesRegex(ValueError, "nb_read_frames"):
            parse_ffprobe_payload(payload)

        payload = make_payload()
        payload["streams"][0]["avg_frame_rate"] = "0/0"
        with self.assertRaisesRegex(ValueError, "avg_frame_rate"):
            parse_ffprobe_payload(payload)


if __name__ == "__main__":
    unittest.main()
