"""GT 없는 2x VFI/VSR 순서 비교용 보수적 temporal/spatial proxy입니다."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np


@dataclass(frozen=True, slots=True)
class VariantQualityMetrics:
    """동일 decode timeline에서 계산한 순서 비교용 집계값입니다."""

    frame_count: int
    source_timestamp_frames: int
    midpoint_frames: int
    source_roundtrip_psnr_db: float
    source_roundtrip_mae: float
    temporal_delta_mae: float
    midpoint_curvature_mae: float
    midpoint_overshoot_rate: float
    laplacian_abs_mean_even: float
    laplacian_abs_mean_odd: float
    laplacian_temporal_delta_mae: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _gray_sample(
    frame: np.ndarray[Any, np.dtype[np.uint8]],
    stride: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    sampled = frame[::stride, ::stride].astype(np.float32)
    return cast(
        np.ndarray[Any, np.dtype[np.float32]],
        sampled[..., 0] * np.float32(0.2126)
        + sampled[..., 1] * np.float32(0.7152)
        + sampled[..., 2] * np.float32(0.0722),
    )


def _laplacian(
    gray: np.ndarray[Any, np.dtype[np.float32]],
) -> np.ndarray[Any, np.dtype[np.float32]]:
    if min(gray.shape) < 3:
        raise ValueError("sampled frame must be at least 3x3")
    return cast(
        np.ndarray[Any, np.dtype[np.float32]],
        np.float32(4.0) * gray[1:-1, 1:-1]
        - gray[:-2, 1:-1]
        - gray[2:, 1:-1]
        - gray[1:-1, :-2]
        - gray[1:-1, 2:],
    )


class OrderQualityAccumulator:
    """2x output을 순차 소비하며 source fidelity와 temporal proxy를 집계합니다."""

    def __init__(
        self,
        source_frames: np.ndarray[Any, np.dtype[np.uint8]],
        *,
        output_width: int,
        output_height: int,
        cut_after: tuple[int, ...],
        sample_stride: int = 2,
    ) -> None:
        if source_frames.dtype != np.uint8:
            raise TypeError("source_frames must be uint8")
        if source_frames.ndim != 4 or source_frames.shape[-1] != 3:
            raise ValueError("source_frames must have shape [N,H,W,3]")
        if source_frames.shape[0] < 2:
            raise ValueError("source_frames must contain at least two frames")
        if (output_width, output_height) != (
            source_frames.shape[2] * 2,
            source_frames.shape[1] * 2,
        ):
            raise ValueError("output geometry must be exactly 2x source geometry")
        if isinstance(sample_stride, bool) or sample_stride < 1:
            raise ValueError("sample_stride must be a positive integer")
        if any(index < 0 or index >= source_frames.shape[0] - 1 for index in cut_after):
            raise ValueError("cut_after contains an invalid source transition")
        if len(set(cut_after)) != len(cut_after):
            raise ValueError("cut_after must contain unique transitions")
        if len(cut_after) >= source_frames.shape[0] - 1:
            raise ValueError("at least one non-cut transition is required")
        self._source = source_frames
        self._width = output_width
        self._height = output_height
        self._frame_bytes = output_width * output_height * 3
        self._expected_frames = source_frames.shape[0] * 2
        self._cuts = frozenset(cut_after)
        self._stride = sample_stride
        self._next_index = 0
        self._source_mse_sum = 0.0
        self._source_mae_sum = 0.0
        self._source_count = 0
        self._temporal_delta_sum = 0.0
        self._temporal_delta_count = 0
        self._curvature_sum = 0.0
        self._curvature_count = 0
        self._overshoot_values = 0
        self._overshoot_count = 0
        self._lap_even_sum = 0.0
        self._lap_even_count = 0
        self._lap_odd_sum = 0.0
        self._lap_odd_count = 0
        self._lap_delta_sum = 0.0
        self._lap_delta_count = 0
        self._previous_gray: np.ndarray[Any, Any] | None = None
        self._previous_laplacian: np.ndarray[Any, Any] | None = None
        self._previous_even: np.ndarray[Any, Any] | None = None
        self._pending_odd: np.ndarray[Any, Any] | None = None

    def consume(self, frame_index: int, frame: bytes) -> None:
        """순차 RGB24 output frame 하나를 집계합니다."""

        if frame_index != self._next_index:
            raise ValueError(
                f"frame index must be sequential: expected={self._next_index}, "
                f"actual={frame_index}"
            )
        if not isinstance(frame, bytes) or len(frame) != self._frame_bytes:
            raise ValueError("frame must be exact RGB24 bytes")
        array = np.frombuffer(frame, dtype=np.uint8).reshape(
            self._height,
            self._width,
            3,
        )
        gray = _gray_sample(array, self._stride)
        laplacian = _laplacian(gray)
        if self._previous_gray is not None:
            self._temporal_delta_sum += float(
                np.mean(np.abs(gray - self._previous_gray), dtype=np.float64) / 255.0
            )
            self._temporal_delta_count += 1
        if self._previous_laplacian is not None:
            self._lap_delta_sum += float(
                np.mean(
                    np.abs(laplacian - self._previous_laplacian),
                    dtype=np.float64,
                )
                / 255.0
            )
            self._lap_delta_count += 1

        if frame_index % 2 == 0:
            source_index = frame_index // 2
            downsampled = array.reshape(
                self._source.shape[1],
                2,
                self._source.shape[2],
                2,
                3,
            ).mean(axis=(1, 3), dtype=np.float32)
            difference = downsampled - self._source[source_index].astype(np.float32)
            self._source_mse_sum += float(np.mean(difference * difference, dtype=np.float64))
            self._source_mae_sum += float(
                np.mean(np.abs(difference), dtype=np.float64) / 255.0
            )
            self._source_count += 1
            self._lap_even_sum += float(
                np.mean(np.abs(laplacian), dtype=np.float64) / 255.0
            )
            self._lap_even_count += 1
            if self._pending_odd is not None and self._previous_even is not None:
                transition_index = source_index - 1
                if transition_index not in self._cuts:
                    midpoint = (self._previous_even + gray) * np.float32(0.5)
                    self._curvature_sum += float(
                        np.mean(
                            np.abs(self._pending_odd - midpoint),
                            dtype=np.float64,
                        )
                        / 255.0
                    )
                    self._curvature_count += 1
                    lower = np.minimum(self._previous_even, gray)
                    upper = np.maximum(self._previous_even, gray)
                    outside = np.logical_or(
                        self._pending_odd < lower,
                        self._pending_odd > upper,
                    )
                    self._overshoot_values += int(np.count_nonzero(outside))
                    self._overshoot_count += outside.size
            self._previous_even = gray.copy()
            self._pending_odd = None
        else:
            self._lap_odd_sum += float(
                np.mean(np.abs(laplacian), dtype=np.float64) / 255.0
            )
            self._lap_odd_count += 1
            self._pending_odd = gray.copy()

        self._previous_gray = gray
        self._previous_laplacian = laplacian
        self._next_index += 1

    def finalize(self) -> VariantQualityMetrics:
        """정확한 frame count와 모든 metric denominator를 검증합니다."""

        if self._next_index != self._expected_frames:
            raise RuntimeError(
                f"output stream ended early: expected={self._expected_frames}, "
                f"actual={self._next_index}"
            )
        expected_midpoints = self._source.shape[0] - 1 - len(self._cuts)
        if self._curvature_count != expected_midpoints:
            raise RuntimeError(
                f"midpoint count mismatch: expected={expected_midpoints}, "
                f"actual={self._curvature_count}"
            )
        if self._source_count != self._source.shape[0]:
            raise RuntimeError("source timestamp frame count mismatch")
        mean_mse = self._source_mse_sum / self._source_count
        psnr = 100.0 if mean_mse == 0.0 else 10.0 * math.log10(255.0**2 / mean_mse)
        return VariantQualityMetrics(
            frame_count=self._next_index,
            source_timestamp_frames=self._source_count,
            midpoint_frames=self._curvature_count,
            source_roundtrip_psnr_db=psnr,
            source_roundtrip_mae=self._source_mae_sum / self._source_count,
            temporal_delta_mae=self._temporal_delta_sum / self._temporal_delta_count,
            midpoint_curvature_mae=self._curvature_sum / self._curvature_count,
            midpoint_overshoot_rate=self._overshoot_values / self._overshoot_count,
            laplacian_abs_mean_even=self._lap_even_sum / self._lap_even_count,
            laplacian_abs_mean_odd=self._lap_odd_sum / self._lap_odd_count,
            laplacian_temporal_delta_mae=(
                self._lap_delta_sum / self._lap_delta_count
            ),
        )
