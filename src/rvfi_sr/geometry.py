"""모델 alignment padding과 정확한 출력 crop 계약을 정의합니다."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AlignedGeometry:
    """입력 해상도를 모델 alignment에 맞추고 정확한 SR 크기를 보존합니다."""

    width: int
    height: int
    scale: int
    alignment: int
    padded_width: int
    padded_height: int

    @classmethod
    def create(
        cls,
        width: int,
        height: int,
        scale: int,
        alignment: int,
    ) -> "AlignedGeometry":
        """기하 파라미터를 검증하고 오른쪽·아래 padding 계획을 생성합니다."""

        for name, value in (("width", width), ("height", height)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if isinstance(scale, bool) or not isinstance(scale, int):
            raise TypeError("scale must be an integer")
        if scale <= 0:
            raise ValueError("scale must be positive")
        if isinstance(alignment, bool) or not isinstance(alignment, int):
            raise TypeError("alignment must be an integer")
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("alignment must be a positive power of two")

        padded_width = ((width + alignment - 1) // alignment) * alignment
        padded_height = ((height + alignment - 1) // alignment) * alignment
        return cls(
            width=width,
            height=height,
            scale=scale,
            alignment=alignment,
            padded_width=padded_width,
            padded_height=padded_height,
        )

    @property
    def pad_right(self) -> int:
        """입력 오른쪽 padding 크기를 반환합니다."""

        return self.padded_width - self.width

    @property
    def pad_bottom(self) -> int:
        """입력 아래쪽 padding 크기를 반환합니다."""

        return self.padded_height - self.height

    @property
    def output_width(self) -> int:
        """padding 제거 후 정확한 SR 출력 너비를 반환합니다."""

        return self.width * self.scale

    @property
    def output_height(self) -> int:
        """padding 제거 후 정확한 SR 출력 높이를 반환합니다."""

        return self.height * self.scale

    @property
    def output_crop_box(self) -> tuple[int, int, int, int]:
        """왼쪽, 위, 오른쪽, 아래 순서의 출력 crop box를 반환합니다."""

        return (0, 0, self.output_width, self.output_height)
