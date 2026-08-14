"""RGB frame stream을 bounded-memory atomic NPY chunk로 조립합니다."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from rvfi_sr.rife_chunks import RifeChunk


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
