"""Hydra payload를 실행 전 엄격하게 검사하는 Pydantic 설정입니다."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rvfi_sr.backends import BackendRequest
from rvfi_sr.registry import get_backend_capabilities


class PipelineOrder(StrEnum):
    """품질 A/B 비교에 사용하는 처리 단계 순서입니다."""

    VFI_THEN_VSR = "vfi_then_vsr"
    VSR_THEN_VFI = "vsr_then_vfi"


class StrictConfigModel(BaseModel):
    """알 수 없는 key와 실행 중 설정 변조를 차단합니다."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class VFIConfig(StrictConfigModel):
    """시간축 보간 backend 설정입니다."""

    backend_id: str
    temporal_multiplier: Annotated[int, Field(ge=2)] = 2


class VSRConfig(StrictConfigModel):
    """공간축 복원 backend 및 명시적 후처리 설정입니다."""

    backend_id: str
    spatial_scale: Annotated[int, Field(ge=2)] = 2
    post_downsample: bool = False


class SceneCutConfig(StrictConfigModel):
    """FFmpeg scdet의 명시적 장면 전환 판정 설정입니다."""

    threshold: Annotated[float, Field(gt=0.0, le=100.0)] = 27.0


class RuntimeConfig(StrictConfigModel):
    """GPU와 라이선스 실행 정책입니다."""

    allow_restricted_license: bool = False
    fp16: bool = True
    gpu_index: Annotated[int, Field(ge=0)] = 0
    worker_timeout_seconds: Annotated[int, Field(ge=60)] = 3_600
    final_output_root: Path | None = None

    @field_validator("final_output_root", mode="after")
    @classmethod
    def validate_final_output_root(cls, value: Path | None) -> Path | None:
        """설정된 경우 최종 산출물 root를 절대 경로로 고정합니다."""

        if value is None:
            return None
        if not value.is_absolute():
            raise ValueError("final_output_root must be absolute")
        return value.resolve(strict=False)


class PipelineConfig(StrictConfigModel):
    """전체 실행 전에 경로와 backend 기능을 교차 검증합니다."""

    input_dir: Path
    output_dir: Path
    order: PipelineOrder = PipelineOrder.VFI_THEN_VSR
    vfi: VFIConfig
    vsr: VSRConfig
    scene_cut: SceneCutConfig = SceneCutConfig()
    runtime: RuntimeConfig = RuntimeConfig()

    @field_validator("input_dir", "output_dir", mode="after")
    @classmethod
    def validate_absolute_path(cls, value: Path) -> Path:
        """cwd 의존적인 상대 경로를 거부하고 alias를 정규화합니다."""

        if not value.is_absolute():
            raise ValueError("path must be absolute")
        return value.resolve(strict=False)

    @field_validator("input_dir", mode="after")
    @classmethod
    def validate_input_directory(cls, value: Path) -> Path:
        """실행 전에 존재하는 디렉터리만 입력으로 허용합니다."""

        if not value.is_dir():
            raise ValueError("input_dir must be an existing directory")
        return value

    @model_validator(mode="after")
    def validate_pipeline_contract(self) -> Self:
        """경로·라이선스·모델 capability를 한 번에 교차 검증합니다."""

        if (
            self.output_dir == self.input_dir
            or self.output_dir.is_relative_to(self.input_dir)
            or self.input_dir.is_relative_to(self.output_dir)
        ):
            raise ValueError("input_dir and output_dir must not overlap")
        final_output_root = self.runtime.final_output_root
        if final_output_root is not None and not (
            self.output_dir == final_output_root
            or self.output_dir.is_relative_to(final_output_root)
        ):
            raise ValueError(
                f"output_dir must be under final_output_root: {final_output_root}"
            )

        vfi_capabilities = get_backend_capabilities(self.vfi.backend_id)
        try:
            vfi_capabilities.validate_request(
                BackendRequest(
                    temporal_multiplier=self.vfi.temporal_multiplier,
                    allow_restricted_license=self.runtime.allow_restricted_license,
                )
            )
        except PermissionError as error:
            raise ValueError(str(error)) from error
        vsr_capabilities = get_backend_capabilities(self.vsr.backend_id)
        try:
            vsr_capabilities.validate_request(
                BackendRequest(
                    spatial_scale=self.vsr.spatial_scale,
                    post_downsample=self.vsr.post_downsample,
                    allow_restricted_license=self.runtime.allow_restricted_license,
                )
            )
        except PermissionError as error:
            raise ValueError(str(error)) from error
        return self
