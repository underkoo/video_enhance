"""Hydra composition 결과를 Pydantic 실행 계약으로 변환합니다."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rvfi_sr.config import PipelineConfig


def load_hydra_config(
    config_dir: Path,
    config_name: str,
    overrides: Sequence[str] = (),
) -> PipelineConfig:
    """preset을 compose하고 interpolation까지 해소한 뒤 엄격하게 검증합니다."""

    if not isinstance(config_dir, Path):
        raise TypeError("config_dir must be pathlib.Path")
    resolved_config_dir = config_dir.resolve(strict=True)
    if not resolved_config_dir.is_dir():
        raise NotADirectoryError(resolved_config_dir)
    if not config_name.strip() or Path(config_name).name != config_name:
        raise ValueError("config_name must be a non-empty basename")
    if any(not isinstance(override, str) or not override.strip() for override in overrides):
        raise TypeError("overrides must contain only non-empty strings")

    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(resolved_config_dir),
        job_name="video_enhance_validate",
    ):
        composed = compose(config_name=config_name, overrides=list(overrides))
    payload = OmegaConf.to_container(
        composed,
        resolve=True,
        throw_on_missing=True,
    )
    if not isinstance(payload, dict):
        raise TypeError("Hydra config root must resolve to a mapping")
    return PipelineConfig.model_validate(cast(dict[str, Any], payload))
