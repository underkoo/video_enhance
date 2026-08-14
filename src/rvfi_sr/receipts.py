"""재개 가능한 NPY artifact의 producer와 content 무결성을 증명합니다."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

import numpy as np

_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    """파일 전체의 SHA-256을 bounded memory로 계산합니다."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_text(payload: str) -> str:
    """canonical producer payload를 receipt용 SHA-256으로 변환합니다."""

    if not isinstance(payload, str) or not payload:
        raise ValueError("payload must be a non-empty string")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def receipt_path(artifact_path: Path) -> Path:
    """artifact basename을 보존하는 sidecar receipt 경로를 반환합니다."""

    if not isinstance(artifact_path, Path):
        raise TypeError("artifact_path must be pathlib.Path")
    return artifact_path.with_name(f"{artifact_path.name}.receipt.json")


@dataclass(frozen=True, slots=True)
class NpyArtifactReceipt:
    """producer fingerprint와 실제 NPY content/shape를 함께 고정합니다."""

    schema_version: int
    producer_sha256: str
    artifact_sha256: str
    frame_count: int
    width: int
    height: int
    dtype: str

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported receipt schema: {self.schema_version}")
        for digest_value, digest_name in (
            (self.producer_sha256, "producer_sha256"),
            (self.artifact_sha256, "artifact_sha256"),
        ):
            if not isinstance(digest_value, str) or not _SHA256_PATTERN.fullmatch(
                digest_value
            ):
                raise ValueError(f"{digest_name} must be a lowercase SHA-256 digest")
        for numeric_value, numeric_name in (
            (self.frame_count, "frame_count"),
            (self.width, "width"),
            (self.height, "height"),
        ):
            if (
                isinstance(numeric_value, bool)
                or not isinstance(numeric_value, int)
                or numeric_value < 1
            ):
                raise ValueError(f"{numeric_name} must be a positive integer")
        if self.dtype != "uint8":
            raise ValueError("only uint8 NPY receipts are supported")

    @classmethod
    def create(cls, artifact_path: Path, *, producer_sha256: str) -> Self:
        """확정된 uint8 `[N,H,W,3]` NPY를 읽어 receipt를 생성합니다."""

        if not _SHA256_PATTERN.fullmatch(producer_sha256):
            raise ValueError("producer_sha256 must be a lowercase SHA-256 digest")
        resolved = artifact_path.resolve(strict=True)
        frames = np.load(resolved, mmap_mode="r", allow_pickle=False)
        if frames.dtype != np.uint8:
            raise TypeError(f"artifact must be uint8, got {frames.dtype}")
        if frames.ndim != 4 or frames.shape[-1] != 3 or min(frames.shape[:3]) < 1:
            raise ValueError("artifact must have shape [N>=1,H>=1,W>=1,3]")
        return cls(
            schema_version=_SCHEMA_VERSION,
            producer_sha256=producer_sha256,
            artifact_sha256=sha256_file(resolved),
            frame_count=int(frames.shape[0]),
            width=int(frames.shape[2]),
            height=int(frames.shape[1]),
            dtype=str(frames.dtype),
        )

    def to_json(self) -> str:
        return json.dumps(
            asdict(self),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, payload: str) -> Self:
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("receipt must be valid JSON") from error
        if not isinstance(value, dict):
            raise TypeError("receipt root must be an object")
        expected = frozenset(cls.__dataclass_fields__)
        actual = frozenset(value)
        if actual != expected:
            raise ValueError(
                f"receipt fields mismatch: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        return cls(**value)

    def write_atomic(self, artifact_path: Path) -> Path:
        """artifact 옆에 receipt를 원자적으로 기록하고 overwrite를 거부합니다."""

        output_path = receipt_path(artifact_path).resolve(strict=False)
        if output_path.exists():
            raise FileExistsError(output_path)
        partial = output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")
        if partial.exists():
            raise FileExistsError(partial)
        serialized = self.to_json() + "\n"
        try:
            with partial.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(partial, output_path)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        return output_path


def validate_npy_receipt(
    artifact_path: Path,
    *,
    producer_sha256: str,
    expected_shape: tuple[int, int, int, int],
) -> NpyArtifactReceipt:
    """sidecar, producer, 실제 digest 및 NPY metadata를 전부 교차 검증합니다."""

    if len(expected_shape) != 4 or expected_shape[-1] != 3:
        raise ValueError("expected_shape must be [N,H,W,3]")
    sidecar = receipt_path(artifact_path).resolve(strict=True)
    receipt = NpyArtifactReceipt.from_json(sidecar.read_text(encoding="utf-8"))
    if receipt.producer_sha256 != producer_sha256:
        raise RuntimeError(
            f"receipt producer mismatch: expected={producer_sha256}, "
            f"actual={receipt.producer_sha256}"
        )
    actual_shape = (receipt.frame_count, receipt.height, receipt.width, 3)
    if actual_shape != expected_shape:
        raise RuntimeError(
            f"receipt shape mismatch: expected={expected_shape}, actual={actual_shape}"
        )
    actual_sha256 = sha256_file(artifact_path)
    if actual_sha256 != receipt.artifact_sha256:
        raise RuntimeError(
            f"receipt artifact SHA-256 mismatch: expected={receipt.artifact_sha256}, "
            f"actual={actual_sha256}"
        )
    frames = np.load(artifact_path.resolve(strict=True), mmap_mode="r", allow_pickle=False)
    if frames.dtype != np.uint8 or frames.shape != expected_shape:
        raise RuntimeError(
            f"NPY metadata mismatch: expected=({expected_shape}, uint8), "
            f"actual=({frames.shape}, {frames.dtype})"
        )
    return receipt


def write_npy_receipt(artifact_path: Path, *, producer_sha256: str) -> Path:
    """NPY receipt를 생성해 atomic sidecar로 확정합니다."""

    receipt = NpyArtifactReceipt.create(
        artifact_path,
        producer_sha256=producer_sha256,
    )
    return receipt.write_atomic(artifact_path)
