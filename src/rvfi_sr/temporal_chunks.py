"""scene 경계를 보존하는 FlashVSR causal temporal chunk 계획입니다."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise

_MODEL_FRAMES = 21
_LEFT_CONTEXT_FRAMES = 5


@dataclass(frozen=True, slots=True)
class TemporalChunk:
    """입력 slice, terminal padding, 최종 출력 ownership을 함께 고정합니다."""

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
        if source_frames + self.pad_terminal != _MODEL_FRAMES:
            raise ValueError("source frames plus terminal padding must equal 21")
        if not (0 <= self.keep_start < self.keep_stop <= source_frames):
            raise ValueError("keep range must be a non-empty subset of decoded source frames")

    @property
    def output_start(self) -> int:
        """전체 영상 좌표계의 첫 소유 출력 index입니다."""

        return self.source_start + self.keep_start

    @property
    def output_stop(self) -> int:
        """전체 영상 좌표계의 exclusive 소유 출력 index입니다."""

        return self.source_start + self.keep_stop


def _scene_segments(frame_count: int, cut_after: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    boundaries = (0, *(index + 1 for index in cut_after), frame_count)
    return tuple(pairwise(boundaries))


def plan_flashvsr_chunks(
    *,
    frame_count: int,
    cut_after: Iterable[int],
) -> tuple[TemporalChunk, ...]:
    """장면별로 21-frame 입력과 5-frame left context ownership을 계획합니다."""

    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise TypeError("frame_count must be an integer")
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    cuts = tuple(cut_after)
    if any(isinstance(index, bool) or not isinstance(index, int) for index in cuts):
        raise TypeError("cut_after indices must be integers")
    if len(set(cuts)) != len(cuts):
        raise ValueError("duplicate cut_after indices are not allowed")
    if any(index < 0 or index >= frame_count - 1 for index in cuts):
        raise ValueError("cut_after indices must be in [0, frame_count - 2]")
    cuts = tuple(sorted(cuts))

    chunks: list[TemporalChunk] = []
    for segment_start, segment_stop in _scene_segments(frame_count, cuts):
        output_cursor = segment_start
        first_chunk = True
        while output_cursor < segment_stop:
            source_start = (
                segment_start
                if first_chunk
                else output_cursor - _LEFT_CONTEXT_FRAMES
            )
            source_stop = min(source_start + _MODEL_FRAMES, segment_stop)
            source_frames = source_stop - source_start
            keep_start = output_cursor - source_start
            keep_stop = source_frames
            chunk = TemporalChunk(
                segment_start=segment_start,
                segment_stop=segment_stop,
                source_start=source_start,
                source_stop=source_stop,
                pad_terminal=_MODEL_FRAMES - source_frames,
                keep_start=keep_start,
                keep_stop=keep_stop,
            )
            chunks.append(chunk)
            output_cursor = chunk.output_stop
            first_chunk = False

    if chunks[0].output_start != 0 or chunks[-1].output_stop != frame_count:
        raise RuntimeError("temporal chunk coverage endpoints are invalid")
    for previous, current in pairwise(chunks):
        if previous.output_stop != current.output_start:
            raise RuntimeError("temporal chunk output ownership has a gap or overlap")
    return tuple(chunks)
