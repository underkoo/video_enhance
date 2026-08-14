"""scene 경계를 보존하는 Practical-RIFE source chunk 계획입니다."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True, slots=True)
class RifeChunk:
    """RIFE worker 입력과 worker 출력 중 소유할 local slice를 고정합니다."""

    segment_start: int
    segment_stop: int
    source_start: int
    source_stop: int
    multiplier: int
    keep_start: int
    keep_stop: int
    use_model: bool

    def __post_init__(self) -> None:
        if not (
            0 <= self.segment_start
            <= self.source_start
            < self.source_stop
            <= self.segment_stop
        ):
            raise ValueError("source range must be a non-empty subset of its scene segment")
        if self.multiplier < 2:
            raise ValueError("multiplier must be at least 2")
        source_frames = self.source_stop - self.source_start
        if self.use_model != (source_frames >= 2):
            raise ValueError("only multi-frame chunks may invoke the RIFE model")
        worker_output_frames = source_frames * self.multiplier
        if not (0 <= self.keep_start < self.keep_stop <= worker_output_frames):
            raise ValueError("keep range must be inside the worker output")

    @property
    def source_frames(self) -> int:
        return self.source_stop - self.source_start

    @property
    def worker_output_frames(self) -> int:
        return self.source_frames * self.multiplier

    @property
    def output_start(self) -> int:
        return self.source_start * self.multiplier + self.keep_start

    @property
    def output_stop(self) -> int:
        return self.source_start * self.multiplier + self.keep_stop


def _scene_segments(frame_count: int, cuts: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    boundaries = (0, *(index + 1 for index in cuts), frame_count)
    return tuple(pairwise(boundaries))


def plan_rife_chunks(
    *,
    frame_count: int,
    cut_after: Iterable[int],
    multiplier: int,
    max_source_frames: int = 64,
) -> tuple[RifeChunk, ...]:
    """scene별 terminal hold와 chunk 경계 interpolation ownership을 계획합니다."""

    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise TypeError("frame_count must be an integer")
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if isinstance(multiplier, bool) or not isinstance(multiplier, int):
        raise TypeError("multiplier must be an integer")
    if multiplier < 2:
        raise ValueError("multiplier must be at least 2")
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

    chunks: list[RifeChunk] = []
    for segment_start, segment_stop in _scene_segments(frame_count, cuts):
        if segment_stop - segment_start == 1:
            chunks.append(
                RifeChunk(
                    segment_start=segment_start,
                    segment_stop=segment_stop,
                    source_start=segment_start,
                    source_stop=segment_stop,
                    multiplier=multiplier,
                    keep_start=0,
                    keep_stop=multiplier,
                    use_model=False,
                )
            )
            continue

        source_start = segment_start
        first_chunk = True
        while source_start < segment_stop - 1:
            source_stop = min(source_start + max_source_frames, segment_stop)
            source_frames = source_stop - source_start
            final_chunk = source_stop == segment_stop
            keep_start = 0 if first_chunk else 1
            keep_stop = (
                source_frames * multiplier
                if final_chunk
                else (source_frames - 1) * multiplier + 1
            )
            chunks.append(
                RifeChunk(
                    segment_start=segment_start,
                    segment_stop=segment_stop,
                    source_start=source_start,
                    source_stop=source_stop,
                    multiplier=multiplier,
                    keep_start=keep_start,
                    keep_stop=keep_stop,
                    use_model=True,
                )
            )
            if final_chunk:
                break
            source_start = source_stop - 1
            first_chunk = False

    if chunks[0].output_start != 0 or chunks[-1].output_stop != frame_count * multiplier:
        raise RuntimeError("RIFE chunk coverage endpoints are invalid")
    for previous, current in pairwise(chunks):
        if previous.output_stop != current.output_start:
            raise RuntimeError("RIFE chunk output ownership has a gap or overlap")
    return tuple(chunks)
