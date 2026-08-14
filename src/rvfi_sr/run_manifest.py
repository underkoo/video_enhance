"""재개 전에 입력·설정·timeline plan 동일성을 검증하는 run manifest입니다."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_FPS_PATTERN = re.compile(r"[1-9][0-9]*/[1-9][0-9]*")


@dataclass(frozen=True, slots=True)
class RunPlanManifest:
    """artifact producer fingerprint의 root가 되는 immutable pipeline plan입니다."""

    schema_version: int
    input_path: str
    output_path: str
    input_sha256: str
    config_sha256: str
    source_width: int
    source_height: int
    source_frames: int
    cfr_frames: int
    cfr_dropped_frames: int
    cfr_duplicated_frames: int
    target_fps: str
    scene_cut_after: tuple[int, ...]
    vfi_backend_id: str
    vfi_multiplier: int
    rife_chunk_count: int
    vsr_backend_id: str
    vsr_scale: int
    vsr_max_model_frames: int
    vsr_chunk_count: int
    output_fps: str
    output_frames: int

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported run manifest schema: {self.schema_version}")
        for path_value, path_name in (
            (self.input_path, "input_path"),
            (self.output_path, "output_path"),
        ):
            if not isinstance(path_value, str) or not Path(path_value).is_absolute():
                raise ValueError(f"{path_name} must be an absolute path string")
        for digest_value, digest_name in (
            (self.input_sha256, "input_sha256"),
            (self.config_sha256, "config_sha256"),
        ):
            if not isinstance(digest_value, str) or not _SHA256_PATTERN.fullmatch(
                digest_value
            ):
                raise ValueError(f"{digest_name} must be a lowercase SHA-256 digest")
        positive_values = (
            (self.source_width, "source_width"),
            (self.source_height, "source_height"),
            (self.source_frames, "source_frames"),
            (self.cfr_frames, "cfr_frames"),
            (self.vfi_multiplier, "vfi_multiplier"),
            (self.rife_chunk_count, "rife_chunk_count"),
            (self.vsr_scale, "vsr_scale"),
            (self.vsr_max_model_frames, "vsr_max_model_frames"),
            (self.vsr_chunk_count, "vsr_chunk_count"),
            (self.output_frames, "output_frames"),
        )
        for positive_value, positive_name in positive_values:
            if (
                isinstance(positive_value, bool)
                or not isinstance(positive_value, int)
                or positive_value < 1
            ):
                raise ValueError(f"{positive_name} must be a positive integer")
        for count_value, count_name in (
            (self.cfr_dropped_frames, "cfr_dropped_frames"),
            (self.cfr_duplicated_frames, "cfr_duplicated_frames"),
        ):
            if (
                isinstance(count_value, bool)
                or not isinstance(count_value, int)
                or count_value < 0
            ):
                raise ValueError(f"{count_name} must be a non-negative integer")
        if not _FPS_PATTERN.fullmatch(self.target_fps) or not _FPS_PATTERN.fullmatch(
            self.output_fps
        ):
            raise ValueError("FPS values must use positive numerator/denominator strings")
        if not self.vfi_backend_id or not self.vsr_backend_id:
            raise ValueError("backend identifiers must not be empty")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in self.scene_cut_after
        ):
            raise ValueError("scene_cut_after must contain non-negative integers")
        if tuple(sorted(set(self.scene_cut_after))) != self.scene_cut_after:
            raise ValueError("scene_cut_after must be sorted and unique")

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
            raise ValueError("run manifest must be valid JSON") from error
        if not isinstance(value, dict):
            raise TypeError("run manifest root must be an object")
        expected = frozenset(cls.__dataclass_fields__)
        actual = frozenset(value)
        if actual != expected:
            raise ValueError(
                f"run manifest fields mismatch: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        cuts = value.get("scene_cut_after")
        if not isinstance(cuts, list):
            raise TypeError("scene_cut_after must be a JSON array")
        value["scene_cut_after"] = tuple(cuts)
        return cls(**value)


def _write_atomic_json(output_path: Path, payload: str) -> Path:
    if not output_path.is_absolute() or output_path.suffix.casefold() != ".json":
        raise ValueError("output_path must be an absolute JSON path")
    resolved = output_path.resolve(strict=False)
    if resolved.exists():
        raise FileExistsError(resolved)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    partial = resolved.with_name(f"{resolved.stem}.partial{resolved.suffix}")
    if partial.exists():
        raise FileExistsError(partial)
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, resolved)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return resolved


def write_run_plan(output_path: Path, plan: RunPlanManifest) -> Path:
    """작업 디렉터리의 immutable plan을 atomic 기록합니다."""

    if not isinstance(plan, RunPlanManifest):
        raise TypeError("plan must be RunPlanManifest")
    return _write_atomic_json(output_path, plan.to_json())


def validate_run_plan(output_path: Path, expected: RunPlanManifest) -> None:
    """재계산한 plan과 저장된 plan의 field-level 동일성을 검사합니다."""

    if not isinstance(expected, RunPlanManifest):
        raise TypeError("expected must be RunPlanManifest")
    actual = RunPlanManifest.from_json(
        output_path.resolve(strict=True).read_text(encoding="utf-8")
    )
    if actual != expected:
        differences = tuple(
            name
            for name in expected.__dataclass_fields__
            if getattr(actual, name) != getattr(expected, name)
        )
        raise RuntimeError(f"resume run plan mismatch: fields={differences}")


def write_completed_run_manifest(
    output_path: Path,
    *,
    plan: RunPlanManifest,
    output_sha256: str,
    output_size_bytes: int,
    elapsed_seconds: float,
) -> Path:
    """검증된 최종 MP4의 digest/size와 실행 plan을 함께 기록합니다."""

    if not _SHA256_PATTERN.fullmatch(output_sha256):
        raise ValueError("output_sha256 must be a lowercase SHA-256 digest")
    if (
        isinstance(output_size_bytes, bool)
        or not isinstance(output_size_bytes, int)
        or output_size_bytes < 1
    ):
        raise ValueError("output_size_bytes must be a positive integer")
    if not isinstance(elapsed_seconds, int | float) or not math.isfinite(
        elapsed_seconds
    ) or elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be finite and positive")
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "plan": asdict(plan),
        "result": {
            "output_sha256": output_sha256,
            "output_size_bytes": output_size_bytes,
            "elapsed_seconds": float(elapsed_seconds),
        },
    }
    return _write_atomic_json(
        output_path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
    )
