"""RTX 3090 실측 기반 RealBasicVSR resolution 및 temporal ownership 계약입니다."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise

_SPYNET_ALIGNMENT = 32
_MIN_MODEL_FRAMES = 2
_MAX_PADDED_FRAME_PIXELS = 2 * 1920 * 1088


@dataclass(frozen=True, slots=True)
class RealBasicVSRResolutionContract:
    """SPyNet 내부 정렬을 포함한 RTX 3090 frame-pixel 한도입니다."""

    width: int
    height: int
    model_frames: int
    padded_width: int
    padded_height: int
    padded_frame_pixels: int

    @classmethod
    def create(
        cls,
        *,
        width: int,
        height: int,
        model_frames: int,
        output_scale: int,
    ) -> RealBasicVSRResolutionContract:
        """실측하지 않은 scale과 RTX 3090 메모리 범위를 사전 거부합니다."""

        for value, name in (
            (width, "width"),
            (height, "height"),
            (model_frames, "model_frames"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if width < 64 or height < 64:
            raise ValueError("RealBasicVSR width and height must be at least 64")
        if model_frames < _MIN_MODEL_FRAMES:
            raise ValueError("RealBasicVSR model_frames must be at least 2")
        if output_scale != 2:
            raise ValueError("only measured RealBasicVSR output_scale=2 is allowed")
        padded_width = ((width + _SPYNET_ALIGNMENT - 1) // _SPYNET_ALIGNMENT) * _SPYNET_ALIGNMENT
        padded_height = (
            (height + _SPYNET_ALIGNMENT - 1) // _SPYNET_ALIGNMENT
        ) * _SPYNET_ALIGNMENT
        padded_frame_pixels = model_frames * padded_width * padded_height
        if padded_frame_pixels > _MAX_PADDED_FRAME_PIXELS:
            raise ValueError(
                "RealBasicVSR request exceeds the measured RTX 3090 limit: "
                f"actual={padded_frame_pixels}, limit={_MAX_PADDED_FRAME_PIXELS}"
            )
        return cls(
            width=width,
            height=height,
            model_frames=model_frames,
            padded_width=padded_width,
            padded_height=padded_height,
            padded_frame_pixels=padded_frame_pixels,
        )

    @classmethod
    def max_model_frames(cls, *, width: int, height: int) -> int:
        """주어진 해상도에서 실측 한도를 넘지 않는 최대 model frame 수입니다."""

        contract = cls.create(
            width=width,
            height=height,
            model_frames=_MIN_MODEL_FRAMES,
            output_scale=2,
        )
        return _MAX_PADDED_FRAME_PIXELS // (
            contract.padded_width * contract.padded_height
        )


@dataclass(frozen=True, slots=True)
class RealBasicVSRChunk:
    """bidirectional context, terminal padding 및 전역 output ownership입니다."""

    segment_start: int
    segment_stop: int
    source_start: int
    source_stop: int
    pad_terminal: int
    keep_start: int
    keep_stop: int

    def __post_init__(self) -> None:
        if not (
            0 <= self.segment_start
            <= self.source_start
            < self.source_stop
            <= self.segment_stop
        ):
            raise ValueError("source range must be a non-empty subset of its scene segment")
        source_frames = self.source_stop - self.source_start
        if source_frames + self.pad_terminal < _MIN_MODEL_FRAMES:
            raise ValueError("source frames plus padding must provide at least two frames")
        if not (0 <= self.keep_start < self.keep_stop <= source_frames):
            raise ValueError("keep range must be inside unpadded source frames")

    @property
    def model_frames(self) -> int:
        return self.source_stop - self.source_start + self.pad_terminal

    @property
    def output_start(self) -> int:
        return self.source_start + self.keep_start

    @property
    def output_stop(self) -> int:
        return self.source_start + self.keep_stop


def _scene_segments(frame_count: int, cuts: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    boundaries = (0, *(index + 1 for index in cuts), frame_count)
    return tuple(pairwise(boundaries))


def plan_realbasicvsr_chunks(
    *,
    frame_count: int,
    cut_after: Iterable[int],
    max_source_frames: int,
) -> tuple[RealBasicVSRChunk, ...]:
    """scene을 넘지 않는 bidirectional context와 exact output coverage를 계획합니다."""

    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise TypeError("frame_count must be an integer")
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if isinstance(max_source_frames, bool) or not isinstance(max_source_frames, int):
        raise TypeError("max_source_frames must be an integer")
    if max_source_frames < 2:
        raise ValueError("max_source_frames must be at least 2")
    cuts = tuple(cut_after)
    if any(isinstance(index, bool) or not isinstance(index, int) for index in cuts):
        raise TypeError("cut_after indices must be integers")
    if len(set(cuts)) != len(cuts):
        raise ValueError("duplicate cut_after indices are not allowed")
    if any(index < 0 or index >= frame_count - 1 for index in cuts):
        raise ValueError("cut_after indices must be in [0, frame_count - 2]")
    cuts = tuple(sorted(cuts))

    chunks: list[RealBasicVSRChunk] = []
    for segment_start, segment_stop in _scene_segments(frame_count, cuts):
        if segment_stop - segment_start == 1:
            chunks.append(
                RealBasicVSRChunk(
                    segment_start=segment_start,
                    segment_stop=segment_stop,
                    source_start=segment_start,
                    source_stop=segment_stop,
                    pad_terminal=1,
                    keep_start=0,
                    keep_stop=1,
                )
            )
            continue

        output_cursor = segment_start
        while output_cursor < segment_stop:
            if max_source_frames == 2:
                source_start = min(output_cursor, segment_stop - 2)
                source_stop = source_start + 2
                final_chunk = source_stop == segment_stop
                keep_start = output_cursor - source_start
                keep_stop = 2 if final_chunk else keep_start + 1
            else:
                source_start = (
                    segment_start
                    if output_cursor == segment_start
                    else output_cursor - 1
                )
                source_stop = min(source_start + max_source_frames, segment_stop)
                final_chunk = source_stop == segment_stop
                keep_start = output_cursor - source_start
                keep_stop = source_stop - source_start
                if not final_chunk:
                    keep_stop -= 1
            chunk = RealBasicVSRChunk(
                segment_start=segment_start,
                segment_stop=segment_stop,
                source_start=source_start,
                source_stop=source_stop,
                pad_terminal=0,
                keep_start=keep_start,
                keep_stop=keep_stop,
            )
            chunks.append(chunk)
            output_cursor = chunk.output_stop

    if chunks[0].output_start != 0 or chunks[-1].output_stop != frame_count:
        raise RuntimeError("RealBasicVSR chunk coverage endpoints are invalid")
    for previous, current in pairwise(chunks):
        if previous.output_stop != current.output_start:
            raise RuntimeError("RealBasicVSR output ownership has a gap or overlap")
    return tuple(chunks)
