"""FlashVSR v1.1 streaming tiny의 시간축 입력 계약입니다."""

from __future__ import annotations

from dataclasses import dataclass

_SOURCE_ALIGNMENT = 32
_RTX_3090_TESTED_PADDED_PIXELS = 608 * 1088


@dataclass(frozen=True, slots=True)
class FlashVSRChunkContract:
    """공식 streaming loop가 모든 입력 프레임을 처리하는 chunk 크기입니다."""

    source_frames: int
    terminal_padding: int = 4

    @classmethod
    def create(cls, source_frames: int) -> FlashVSRChunkContract:
        """`8n-3` 및 최소 loop 횟수를 검증합니다."""

        if isinstance(source_frames, bool) or not isinstance(source_frames, int):
            raise TypeError("source_frames must be an integer")
        if source_frames < 21:
            raise ValueError("FlashVSR source_frames must be at least 21")
        if (source_frames + 3) % 8 != 0:
            raise ValueError("FlashVSR source_frames must have the form 8n - 3")
        return cls(source_frames=source_frames)

    @property
    def padded_frames(self) -> int:
        """공식 구현에 전달할 terminal-repeat 포함 frame 수를 반환합니다."""

        return self.source_frames + self.terminal_padding

    @property
    def process_iterations(self) -> int:
        """공식 tiny pipeline 내부 streaming iteration 수를 반환합니다."""

        return (self.padded_frames - 1) // 8 - 2


@dataclass(frozen=True, slots=True)
class FlashVSRResolutionContract:
    """RTX 3090에서 실측 검증한 single-pass 공간 해상도 계약입니다."""

    width: int
    height: int
    padded_width: int
    padded_height: int

    @classmethod
    def create(cls, width: int, height: int) -> FlashVSRResolutionContract:
        """32 alignment 후 실측 pixel limit을 초과하는 입력을 거부합니다."""

        for name, value in (("width", width), ("height", height)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        padded_width = ((width + _SOURCE_ALIGNMENT - 1) // _SOURCE_ALIGNMENT) * _SOURCE_ALIGNMENT
        padded_height = (
            (height + _SOURCE_ALIGNMENT - 1) // _SOURCE_ALIGNMENT
        ) * _SOURCE_ALIGNMENT
        padded_pixels = padded_width * padded_height
        if padded_pixels > _RTX_3090_TESTED_PADDED_PIXELS:
            raise ValueError(
                "FlashVSR RTX 3090 single-pass limit exceeded: "
                f"padded={padded_width}x{padded_height} ({padded_pixels} pixels), "
                f"tested_limit={_RTX_3090_TESTED_PADDED_PIXELS} pixels"
            )
        return cls(
            width=width,
            height=height,
            padded_width=padded_width,
            padded_height=padded_height,
        )

    @property
    def padded_pixels(self) -> int:
        """alignment 적용 후 모델 입력 pixel 수를 반환합니다."""

        return self.padded_width * self.padded_height
