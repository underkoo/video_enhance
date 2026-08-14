"""격리 모델 백엔드의 기능과 실행 허용 범위를 정의합니다."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class BackendKind(StrEnum):
    """모델 백엔드가 수행하는 시간축 또는 공간축 연산입니다."""

    VFI = "vfi"
    VSR = "vsr"


class LicenseUse(StrEnum):
    """자동 실행 정책에 필요한 라이선스 사용 범위입니다."""

    PERMISSIVE = "permissive"
    NON_COMMERCIAL = "non_commercial"
    RESEARCH_ONLY = "research_only"


@dataclass(frozen=True, slots=True)
class BackendRequest:
    """백엔드 기능 검증에 사용하는 모델 독립 요청입니다."""

    temporal_multiplier: int | None = None
    spatial_scale: int | None = None
    post_downsample: bool = False
    allow_restricted_license: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("temporal_multiplier", self.temporal_multiplier),
            ("spatial_scale", self.spatial_scale),
        ):
            if value is not None and (isinstance(value, bool) or value < 2):
                raise ValueError(f"{name} must be an integer >= 2")


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """오케스트레이터가 모델 import 없이 검사할 수 있는 기능 계약입니다."""

    backend_id: str
    kind: BackendKind
    license_use: LicenseUse
    python_spec: str
    alignment: int
    temporal_multipliers: tuple[int, ...] = ()
    spatial_scales: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", self.backend_id):
            raise ValueError("backend_id must be a lowercase stable identifier")
        if not self.python_spec.strip():
            raise ValueError("python_spec must not be empty")
        if (
            isinstance(self.alignment, bool)
            or self.alignment < 1
            or self.alignment & (self.alignment - 1)
        ):
            raise ValueError("alignment must be a positive power of two")
        self._validate_positive_unique("temporal_multipliers", self.temporal_multipliers)
        self._validate_positive_unique("spatial_scales", self.spatial_scales)

        if self.kind is BackendKind.VFI:
            if not self.temporal_multipliers:
                raise ValueError("VFI backend must declare temporal multipliers")
            if self.spatial_scales:
                raise ValueError("VFI backend must not declare spatial scales")
        elif self.kind is BackendKind.VSR:
            if not self.spatial_scales:
                raise ValueError("VSR backend must declare spatial scales")
            if self.temporal_multipliers:
                raise ValueError("VSR backend must not declare temporal multipliers")
        else:  # pragma: no cover - Enum typing prevents this in normal construction.
            raise TypeError("kind must be BackendKind")

    @staticmethod
    def _validate_positive_unique(name: str, values: tuple[int, ...]) -> None:
        if not isinstance(values, tuple):
            raise TypeError(f"{name} must be a tuple")
        if any(isinstance(value, bool) or value < 2 for value in values):
            raise ValueError(f"{name} values must be integers >= 2")
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must not contain duplicates")
        if tuple(sorted(values)) != values:
            raise ValueError(f"{name} must be sorted")

    def validate_request(self, request: BackendRequest) -> None:
        """요청이 기능 및 라이선스 정책을 모두 만족하는지 검사합니다."""

        if not isinstance(request, BackendRequest):
            raise TypeError("request must be BackendRequest")
        if (
            self.license_use is not LicenseUse.PERMISSIVE
            and not request.allow_restricted_license
        ):
            raise PermissionError(
                f"backend {self.backend_id!r} has a restricted license; "
                "set allow_restricted_license explicitly"
            )

        if self.kind is BackendKind.VFI:
            self._validate_vfi_request(request)
        else:
            self._validate_vsr_request(request)

    def _validate_vfi_request(self, request: BackendRequest) -> None:
        if request.spatial_scale is not None:
            raise ValueError("VFI request must not set spatial_scale")
        if request.post_downsample:
            raise ValueError("VFI request must not enable post_downsample")
        if request.temporal_multiplier not in self.temporal_multipliers:
            raise ValueError(
                f"unsupported temporal multiplier {request.temporal_multiplier!r}; "
                f"supported={self.temporal_multipliers}"
            )

    def _validate_vsr_request(self, request: BackendRequest) -> None:
        if request.temporal_multiplier is not None:
            raise ValueError("VSR request must not set temporal_multiplier")
        if request.spatial_scale in self.spatial_scales:
            if request.post_downsample:
                raise ValueError("post_downsample must be false for a native scale")
            return
        if not request.post_downsample:
            raise ValueError(
                f"spatial scale {request.spatial_scale!r} is not native; "
                "post_downsample must be explicitly enabled"
            )
        requested_scale = request.spatial_scale
        assert requested_scale is not None
        candidates = [
            native_scale
            for native_scale in self.spatial_scales
            if native_scale > requested_scale and native_scale % requested_scale == 0
        ]
        if not candidates:
            raise ValueError(
                f"no native spatial scale has an integer ratio to {requested_scale}; "
                f"supported={self.spatial_scales}"
            )
