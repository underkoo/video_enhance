"""입출력 경로와 원자적 산출물 정책을 정의합니다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ArtifactContract:
    """입력 덮어쓰기를 금지하고 partial 파일을 통한 원자적 완료를 강제합니다."""

    input_path: Path
    output_path: Path
    partial_path: Path

    @classmethod
    def create(cls, input_path: Path, output_path: Path) -> "ArtifactContract":
        """경로를 정규화하고 안전한 산출물 계약을 생성합니다."""

        if not isinstance(input_path, Path) or not isinstance(output_path, Path):
            raise TypeError("input_path and output_path must be pathlib.Path")
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        if output_path.suffix.casefold() != ".mp4":
            raise ValueError("output_path must use the .mp4 suffix")

        resolved_input = input_path.resolve(strict=True)
        resolved_output = output_path.resolve(strict=False)
        if resolved_input == resolved_output:
            raise ValueError("input_path and output_path must differ")

        partial_path = resolved_output.with_name(
            f"{resolved_output.stem}.partial{resolved_output.suffix}"
        )
        return cls(
            input_path=resolved_input,
            output_path=resolved_output,
            partial_path=partial_path,
        )
