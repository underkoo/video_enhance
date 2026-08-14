"""RGB frame stream을 bounded-memory atomic NPY chunk로 조립합니다."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from rvfi_sr.rife_chunks import RifeChunk
from rvfi_sr.temporal_chunks import TemporalChunk


class RifeInputAssembler:
    """순차 RGB24 frame을 겹치는 RIFE 입력 chunk에 정확히 한 번 기록합니다."""

    def __init__(
        self,
        chunks: tuple[RifeChunk, ...],
        *,
        width: int,
        height: int,
        output_dir: Path,
    ) -> None:
        if not chunks:
            raise ValueError("chunks must not be empty")
        if any(not isinstance(chunk, RifeChunk) for chunk in chunks):
            raise TypeError("chunks must contain only RifeChunk values")
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or isinstance(height, bool)
            or not isinstance(height, int)
        ):
            raise TypeError("width and height must be integers")
        if width < 1 or height < 1:
            raise ValueError("width and height must be positive")
        if not isinstance(output_dir, Path) or not output_dir.is_absolute():
            raise ValueError("output_dir must be an absolute pathlib.Path")
        resolved_output_dir = output_dir.resolve(strict=False)
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

        self._chunks = chunks
        self._width = width
        self._height = height
        self._frame_bytes = width * height * 3
        self._frame_count = max(chunk.segment_stop for chunk in chunks)
        self._output_paths = tuple(
            resolved_output_dir / f"rife-input-{index:06d}.npy"
            for index in range(len(chunks))
        )
        for output_path in self._output_paths:
            partial_path = output_path.with_name(
                f"{output_path.stem}.partial{output_path.suffix}"
            )
            if output_path.exists() or partial_path.exists():
                raise FileExistsError(output_path if output_path.exists() else partial_path)
        self._starts: dict[int, list[int]] = {}
        self._stops: dict[int, list[int]] = {}
        for chunk_index, chunk in enumerate(chunks):
            self._starts.setdefault(chunk.source_start, []).append(chunk_index)
            self._stops.setdefault(chunk.source_stop - 1, []).append(chunk_index)
        self._active: dict[int, np.ndarray[Any, np.dtype[np.uint8]]] = {}
        self._next_frame_index = 0
        self._finalized = False

    @property
    def output_paths(self) -> tuple[Path, ...]:
        return self._output_paths

    def consume(self, frame_index: int, frame: bytes) -> None:
        """다음 순차 frame을 active chunk들에 복사하고 완성 chunk를 atomic 확정합니다."""

        if self._finalized:
            raise RuntimeError("assembler is already finalized")
        if frame_index != self._next_frame_index:
            raise ValueError(
                f"frame index must be sequential: expected={self._next_frame_index}, "
                f"actual={frame_index}"
            )
        if not isinstance(frame, bytes):
            raise TypeError("frame must be bytes")
        if len(frame) != self._frame_bytes:
            raise ValueError(
                f"RGB24 frame byte count mismatch: expected={self._frame_bytes}, "
                f"actual={len(frame)}"
            )
        for chunk_index in self._starts.get(frame_index, []):
            chunk = self._chunks[chunk_index]
            self._active[chunk_index] = np.empty(
                (chunk.source_frames, self._height, self._width, 3),
                dtype=np.uint8,
            )
        frame_array = np.frombuffer(frame, dtype=np.uint8).reshape(
            self._height,
            self._width,
            3,
        )
        matching = 0
        for chunk_index, array in self._active.items():
            chunk = self._chunks[chunk_index]
            if chunk.source_start <= frame_index < chunk.source_stop:
                array[frame_index - chunk.source_start] = frame_array
                matching += 1
        if matching < 1:
            raise RuntimeError(f"frame {frame_index} is not owned by any input chunk")
        for chunk_index in self._stops.get(frame_index, []):
            array = self._active.pop(chunk_index)
            self._write_atomic(self._output_paths[chunk_index], array)
        self._next_frame_index += 1

    def finalize(self) -> tuple[Path, ...]:
        """전체 frame/chunk 완성을 검증하고 확정된 경로를 반환합니다."""

        if self._finalized:
            raise RuntimeError("assembler is already finalized")
        if self._next_frame_index != self._frame_count:
            raise RuntimeError(
                f"frame stream ended early: expected={self._frame_count}, "
                f"actual={self._next_frame_index}"
            )
        if self._active:
            raise RuntimeError(f"unfinished chunks remain: {sorted(self._active)}")
        missing = tuple(path for path in self._output_paths if not path.is_file())
        if missing:
            raise RuntimeError(f"chunk artifacts are missing: {missing}")
        self._finalized = True
        return self._output_paths

    @staticmethod
    def _write_atomic(
        output_path: Path,
        frames: np.ndarray[Any, np.dtype[np.uint8]],
    ) -> None:
        partial_path = output_path.with_name(
            f"{output_path.stem}.partial{output_path.suffix}"
        )
        try:
            with partial_path.open("xb") as stream:
                np.save(stream, frames, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(partial_path, output_path)
        except BaseException:
            partial_path.unlink(missing_ok=True)
            raise


def stream_rife_output_frames(
    chunks: tuple[RifeChunk, ...],
    input_paths: tuple[Path, ...],
    worker_output_paths: tuple[Path | None, ...],
    *,
    width: int,
    height: int,
    consume_frame: Callable[[int, bytes], None],
) -> None:
    """RIFE ownership slice와 single-frame bypass를 하나의 연속 RGB stream으로 병합합니다."""

    if not chunks:
        raise ValueError("chunks must not be empty")
    if len(input_paths) != len(chunks) or len(worker_output_paths) != len(chunks):
        raise ValueError("chunk and artifact path counts must match")
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or isinstance(height, bool)
        or not isinstance(height, int)
    ):
        raise TypeError("width and height must be integers")
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    if not callable(consume_frame):
        raise TypeError("consume_frame must be callable")

    output_index = 0
    for chunk, input_path, worker_output_path in zip(
        chunks,
        input_paths,
        worker_output_paths,
        strict=True,
    ):
        if chunk.output_start != output_index:
            raise RuntimeError(
                f"RIFE chunk ownership is discontinuous at output {output_index}"
            )
        if not isinstance(input_path, Path):
            raise TypeError("input_paths must contain pathlib.Path values")
        input_frames = np.load(input_path.resolve(strict=True), mmap_mode="r", allow_pickle=False)
        _validate_uint8_frames(
            input_frames,
            expected_frames=chunk.source_frames,
            width=width,
            height=height,
            artifact_name="RIFE input",
        )
        if chunk.use_model:
            if not isinstance(worker_output_path, Path):
                raise ValueError("model RIFE chunks require a worker output path")
            output_frames = np.load(
                worker_output_path.resolve(strict=True),
                mmap_mode="r",
                allow_pickle=False,
            )
            _validate_uint8_frames(
                output_frames,
                expected_frames=chunk.worker_output_frames,
                width=width,
                height=height,
                artifact_name="RIFE worker output",
            )
            for local_index in range(chunk.keep_start, chunk.keep_stop):
                consume_frame(output_index, output_frames[local_index].tobytes(order="C"))
                output_index += 1
        else:
            if worker_output_path is not None:
                raise ValueError("single-frame bypass chunks must not have a worker output")
            if chunk.source_frames != 1:
                raise RuntimeError("RIFE bypass is only valid for a one-frame scene")
            frame = input_frames[0].tobytes(order="C")
            for _ in range(chunk.keep_start, chunk.keep_stop):
                consume_frame(output_index, frame)
                output_index += 1
    if output_index != chunks[-1].output_stop:
        raise RuntimeError(
            f"RIFE merged output count mismatch: expected={chunks[-1].output_stop}, "
            f"actual={output_index}"
        )


class FlashVSRInputAssembler:
    """연속 RIFE 출력을 21-frame FlashVSR 입력과 terminal padding으로 조립합니다."""

    def __init__(
        self,
        chunks: tuple[TemporalChunk, ...],
        *,
        width: int,
        height: int,
        output_dir: Path,
    ) -> None:
        if not chunks:
            raise ValueError("chunks must not be empty")
        if any(not isinstance(chunk, TemporalChunk) for chunk in chunks):
            raise TypeError("chunks must contain only TemporalChunk values")
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or isinstance(height, bool)
            or not isinstance(height, int)
        ):
            raise TypeError("width and height must be integers")
        if width < 1 or height < 1:
            raise ValueError("width and height must be positive")
        if not isinstance(output_dir, Path) or not output_dir.is_absolute():
            raise ValueError("output_dir must be an absolute pathlib.Path")
        resolved_output_dir = output_dir.resolve(strict=False)
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

        self._chunks = chunks
        self._width = width
        self._height = height
        self._frame_bytes = width * height * 3
        self._frame_count = max(chunk.segment_stop for chunk in chunks)
        self._output_paths = tuple(
            resolved_output_dir / f"flashvsr-input-{index:06d}.npy"
            for index in range(len(chunks))
        )
        for output_path in self._output_paths:
            partial_path = output_path.with_name(
                f"{output_path.stem}.partial{output_path.suffix}"
            )
            if output_path.exists() or partial_path.exists():
                raise FileExistsError(output_path if output_path.exists() else partial_path)
        self._starts: dict[int, list[int]] = {}
        self._stops: dict[int, list[int]] = {}
        for chunk_index, chunk in enumerate(chunks):
            self._starts.setdefault(chunk.source_start, []).append(chunk_index)
            self._stops.setdefault(chunk.source_stop - 1, []).append(chunk_index)
        self._active: dict[int, np.ndarray[Any, np.dtype[np.uint8]]] = {}
        self._next_frame_index = 0
        self._finalized = False

    @property
    def output_paths(self) -> tuple[Path, ...]:
        return self._output_paths

    def consume(self, frame_index: int, frame: bytes) -> None:
        """다음 순차 frame을 모든 active FlashVSR input에 기록합니다."""

        if self._finalized:
            raise RuntimeError("assembler is already finalized")
        if frame_index != self._next_frame_index:
            raise ValueError(
                f"frame index must be sequential: expected={self._next_frame_index}, "
                f"actual={frame_index}"
            )
        if not isinstance(frame, bytes):
            raise TypeError("frame must be bytes")
        if len(frame) != self._frame_bytes:
            raise ValueError(
                f"RGB24 frame byte count mismatch: expected={self._frame_bytes}, "
                f"actual={len(frame)}"
            )
        for chunk_index in self._starts.get(frame_index, []):
            self._active[chunk_index] = np.empty(
                (21, self._height, self._width, 3),
                dtype=np.uint8,
            )
        frame_array = np.frombuffer(frame, dtype=np.uint8).reshape(
            self._height,
            self._width,
            3,
        )
        matching = 0
        for chunk_index, array in self._active.items():
            chunk = self._chunks[chunk_index]
            if chunk.source_start <= frame_index < chunk.source_stop:
                array[frame_index - chunk.source_start] = frame_array
                matching += 1
        if matching < 1:
            raise RuntimeError(f"frame {frame_index} is not owned by any FlashVSR input chunk")
        for chunk_index in self._stops.get(frame_index, []):
            array = self._active.pop(chunk_index)
            chunk = self._chunks[chunk_index]
            source_frames = chunk.source_stop - chunk.source_start
            if chunk.pad_terminal:
                array[source_frames:] = array[source_frames - 1]
            self._write_atomic(self._output_paths[chunk_index], array)
        self._next_frame_index += 1

    def finalize(self) -> tuple[Path, ...]:
        """전체 interpolated timeline과 모든 FlashVSR input artifact를 검증합니다."""

        if self._finalized:
            raise RuntimeError("assembler is already finalized")
        if self._next_frame_index != self._frame_count:
            raise RuntimeError(
                f"frame stream ended early: expected={self._frame_count}, "
                f"actual={self._next_frame_index}"
            )
        if self._active:
            raise RuntimeError(f"unfinished chunks remain: {sorted(self._active)}")
        missing = tuple(path for path in self._output_paths if not path.is_file())
        if missing:
            raise RuntimeError(f"chunk artifacts are missing: {missing}")
        self._finalized = True
        return self._output_paths

    @staticmethod
    def _write_atomic(
        output_path: Path,
        frames: np.ndarray[Any, np.dtype[np.uint8]],
    ) -> None:
        RifeInputAssembler._write_atomic(output_path, frames)


def _validate_uint8_frames(
    frames: np.ndarray[Any, Any],
    *,
    expected_frames: int,
    width: int,
    height: int,
    artifact_name: str,
) -> None:
    if frames.dtype != np.uint8:
        raise TypeError(f"{artifact_name} must be uint8, got {frames.dtype}")
    expected_shape = (expected_frames, height, width, 3)
    if frames.shape != expected_shape:
        raise ValueError(
            f"{artifact_name} shape mismatch: expected={expected_shape}, actual={frames.shape}"
        )


def stream_flashvsr_output_frames(
    chunks: tuple[TemporalChunk, ...],
    worker_output_paths: tuple[Path, ...],
    *,
    width: int,
    height: int,
    consume_frame: Callable[[int, bytes], None],
) -> None:
    """FlashVSR context/padding output을 제거하고 전역 ownership만 순차 방출합니다."""

    if not chunks:
        raise ValueError("chunks must not be empty")
    if len(worker_output_paths) != len(chunks):
        raise ValueError("chunk and worker output path counts must match")
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or isinstance(height, bool)
        or not isinstance(height, int)
    ):
        raise TypeError("width and height must be integers")
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    if not callable(consume_frame):
        raise TypeError("consume_frame must be callable")

    output_index = 0
    for chunk, worker_output_path in zip(chunks, worker_output_paths, strict=True):
        if chunk.output_start != output_index:
            raise RuntimeError(
                f"FlashVSR chunk ownership is discontinuous at output {output_index}"
            )
        if not isinstance(worker_output_path, Path):
            raise TypeError("worker_output_paths must contain pathlib.Path values")
        frames = np.load(
            worker_output_path.resolve(strict=True),
            mmap_mode="r",
            allow_pickle=False,
        )
        _validate_uint8_frames(
            frames,
            expected_frames=21,
            width=width,
            height=height,
            artifact_name="FlashVSR worker output",
        )
        for local_index in range(chunk.keep_start, chunk.keep_stop):
            consume_frame(output_index, frames[local_index].tobytes(order="C"))
            output_index += 1
    if output_index != chunks[-1].output_stop:
        raise RuntimeError(
            f"FlashVSR merged output count mismatch: expected={chunks[-1].output_stop}, "
            f"actual={output_index}"
        )
