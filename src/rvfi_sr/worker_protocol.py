"""제어 환경과 격리 모델 환경 사이의 엄격한 JSON 프로토콜입니다."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

_PROTOCOL_VERSION = 1
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class WorkerStatus(StrEnum):
    """모델 워커의 terminal 상태입니다."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


def _load_object(payload: str, expected_fields: frozenset[str]) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("payload must be valid JSON") from error
    if not isinstance(value, dict):
        raise TypeError("payload root must be a JSON object")
    actual_fields = frozenset(value)
    unexpected = actual_fields - expected_fields
    missing = expected_fields - actual_fields
    if unexpected:
        raise ValueError(f"unexpected fields: {sorted(unexpected)}")
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    return value


def _validate_absolute_distinct_paths(input_path: Path, output_path: Path) -> None:
    if not input_path.is_absolute() or not output_path.is_absolute():
        raise ValueError("input_path and output_path must be absolute")
    if input_path.resolve(strict=False) == output_path.resolve(strict=False):
        raise ValueError("input_path and output_path must differ")


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    """모델별 격리 프로세스에 전달하는 단일 작업 요청입니다."""

    schema_version: int
    job_id: str
    backend_id: str
    input_path: str
    output_path: str
    parameters: dict[str, bool | int | float | str]

    def __post_init__(self) -> None:
        if self.schema_version != _PROTOCOL_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        if not self.job_id.strip():
            raise ValueError("job_id must not be empty")
        if not self.backend_id.strip():
            raise ValueError("backend_id must not be empty")
        _validate_absolute_distinct_paths(Path(self.input_path), Path(self.output_path))
        if not isinstance(self.parameters, dict):
            raise TypeError("parameters must be a JSON object")
        for key, value in self.parameters.items():
            if not isinstance(key, str) or not key:
                raise TypeError("parameter keys must be non-empty strings")
            if not isinstance(value, bool | int | float | str):
                raise TypeError(f"parameter {key!r} is not a scalar JSON value")

    @classmethod
    def create(
        cls,
        *,
        job_id: str,
        backend_id: str,
        input_path: Path,
        output_path: Path,
        parameters: dict[str, bool | int | float | str],
    ) -> Self:
        """경로 별칭을 제거하고 현재 버전의 요청을 생성합니다."""

        if not isinstance(input_path, Path) or not isinstance(output_path, Path):
            raise TypeError("input_path and output_path must be pathlib.Path")
        _validate_absolute_distinct_paths(input_path, output_path)
        return cls(
            schema_version=_PROTOCOL_VERSION,
            job_id=job_id,
            backend_id=backend_id,
            input_path=str(input_path.resolve(strict=False)),
            output_path=str(output_path.resolve(strict=False)),
            parameters=dict(parameters),
        )

    def to_json(self) -> str:
        """요청을 안정적인 단일 JSON 객체로 직렬화합니다."""

        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> Self:
        """알 수 없는 필드까지 거부하며 요청을 역직렬화합니다."""

        value = _load_object(payload, frozenset(cls.__dataclass_fields__))
        return cls(**value)


@dataclass(frozen=True, slots=True)
class WorkerResponse:
    """worker 산출물의 shape 및 digest를 포함하는 terminal 응답입니다."""

    schema_version: int
    job_id: str
    status: WorkerStatus
    output_sha256: str | None
    frame_count: int | None
    width: int | None
    height: int | None
    dtype: str | None

    def __post_init__(self) -> None:
        if self.schema_version != _PROTOCOL_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        if not self.job_id.strip():
            raise ValueError("job_id must not be empty")
        if not isinstance(self.status, WorkerStatus):
            object.__setattr__(self, "status", WorkerStatus(self.status))
        if self.status is WorkerStatus.SUCCEEDED:
            if self.output_sha256 is None or not _SHA256_PATTERN.fullmatch(
                self.output_sha256
            ):
                raise ValueError("successful response requires a valid output_sha256")
            dimensions = (self.frame_count, self.width, self.height)
            if any(
                value is None or isinstance(value, bool) or value < 1
                for value in dimensions
            ):
                raise ValueError("successful response requires positive frame shape")
            if self.dtype not in {"uint8", "uint16", "float16", "float32"}:
                raise ValueError("successful response has an unsupported dtype")
        elif any(
            value is not None
            for value in (
                self.output_sha256,
                self.frame_count,
                self.width,
                self.height,
                self.dtype,
            )
        ):
            raise ValueError("failed response must not claim an output artifact")

    def to_json(self) -> str:
        """응답을 안정적인 단일 JSON 객체로 직렬화합니다."""

        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> Self:
        """알 수 없는 필드까지 거부하며 응답을 역직렬화합니다."""

        value = _load_object(payload, frozenset(cls.__dataclass_fields__))
        return cls(**value)
