"""태그 없는 YUV 입력을 명시적 RGB decode 정책으로 해석합니다."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from rvfi_sr.probe import ColorMetadata


class VideoColorRange(StrEnum):
    """FFmpeg가 이해하는 source sample range입니다."""

    TV = "tv"
    PC = "pc"


class VideoColorSpace(StrEnum):
    """현재 데이터셋에서 허용하는 YUV matrix입니다."""

    BT709 = "bt709"


@dataclass(frozen=True, slots=True)
class RgbDecodeContract:
    """실제 tag와 명시적 untagged 가정을 모두 기록합니다."""

    source_pixel_format: str
    source_range: VideoColorRange
    source_space: VideoColorSpace
    assumed_range: bool
    assumed_space: bool

    @classmethod
    def create(
        cls,
        metadata: ColorMetadata,
        *,
        untagged_range: VideoColorRange,
        untagged_space: VideoColorSpace,
    ) -> RgbDecodeContract:
        """지원 범위 밖 tag를 거부하고 실제 decode matrix/range를 확정합니다."""

        if not isinstance(metadata, ColorMetadata):
            raise TypeError("metadata must be ColorMetadata")
        if not isinstance(untagged_range, VideoColorRange):
            raise TypeError("untagged_range must be VideoColorRange")
        if not isinstance(untagged_space, VideoColorSpace):
            raise TypeError("untagged_space must be VideoColorSpace")
        if metadata.pixel_format != "yuv420p":
            raise ValueError(
                f"unsupported source pixel format: {metadata.pixel_format!r}"
            )
        try:
            source_range = (
                untagged_range
                if metadata.range is None
                else VideoColorRange(metadata.range)
            )
        except ValueError as error:
            raise ValueError(f"unsupported source color range: {metadata.range!r}") from error
        try:
            source_space = (
                untagged_space
                if metadata.space is None
                else VideoColorSpace(metadata.space)
            )
        except ValueError as error:
            raise ValueError(f"unsupported source color space: {metadata.space!r}") from error
        return cls(
            source_pixel_format=metadata.pixel_format,
            source_range=source_range,
            source_space=source_space,
            assumed_range=metadata.range is None,
            assumed_space=metadata.space is None,
        )

    @property
    def ffmpeg_filter(self) -> str:
        """YUV range/matrix를 명시한 full-range RGB24 conversion filter입니다."""

        return (
            f"scale=in_range={self.source_range.value}:"
            f"in_color_matrix={self.source_space.value}:out_range=pc,"
            "format=pix_fmts=rgb24"
        )
