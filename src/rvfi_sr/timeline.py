"""CFR 출력 타임라인과 scene-cut 전환 계약을 정의합니다."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Iterable


class SceneCutPolicy(str, Enum):
    """장면 전환 구간의 중간 timestamp를 채우는 정책입니다."""

    HOLD_PREVIOUS = "hold_previous"


class TransitionKind(str, Enum):
    """인접 원본 프레임 사이에 적용할 작업입니다."""

    INTERPOLATE = "interpolate"
    HOLD_PREVIOUS = "hold_previous"


@dataclass(frozen=True, slots=True)
class TransitionPlan:
    """한 쌍의 원본 프레임 사이에 생성할 프레임 계획입니다."""

    left_index: int
    right_index: int
    kind: TransitionKind
    generated_frames: int


@dataclass(frozen=True, slots=True)
class TimelineContract:
    """입출력 FPS, 프레임 수 및 장면 전환 규칙을 고정합니다."""

    input_frames: int
    input_fps: Fraction
    multiplier: int
    cut_after: tuple[int, ...]
    scene_cut_policy: SceneCutPolicy

    @classmethod
    def create(
        cls,
        input_frames: int,
        input_fps: Fraction,
        multiplier: int,
        cut_after: Iterable[int],
        scene_cut_policy: SceneCutPolicy = SceneCutPolicy.HOLD_PREVIOUS,
    ) -> "TimelineContract":
        """입력을 검증하고 불변 타임라인 계약을 생성합니다."""

        if isinstance(input_frames, bool) or not isinstance(input_frames, int):
            raise TypeError("input_frames must be an integer")
        if input_frames < 2:
            raise ValueError("input_frames must be at least 2")
        if not isinstance(input_fps, Fraction):
            raise TypeError("input_fps must be fractions.Fraction")
        if input_fps <= 0:
            raise ValueError("input_fps must be positive")
        if isinstance(multiplier, bool) or not isinstance(multiplier, int):
            raise TypeError("multiplier must be an integer")
        if multiplier < 2:
            raise ValueError("multiplier must be at least 2")
        if not isinstance(scene_cut_policy, SceneCutPolicy):
            raise TypeError("scene_cut_policy must be SceneCutPolicy")

        cut_indices = tuple(cut_after)
        if any(isinstance(index, bool) or not isinstance(index, int) for index in cut_indices):
            raise TypeError("cut_after indices must be integers")
        if len(set(cut_indices)) != len(cut_indices):
            raise ValueError("duplicate scene-cut indices are not allowed")
        if any(index < 0 or index >= input_frames - 1 for index in cut_indices):
            raise ValueError("cut_after indices must be in [0, input_frames - 2]")

        return cls(
            input_frames=input_frames,
            input_fps=input_fps,
            multiplier=multiplier,
            cut_after=tuple(sorted(cut_indices)),
            scene_cut_policy=scene_cut_policy,
        )

    @property
    def output_fps(self) -> Fraction:
        return self.input_fps * self.multiplier

    @property
    def output_frames(self) -> int:
        return self.input_frames * self.multiplier

    @property
    def terminal_hold_frames(self) -> int:
        return self.multiplier - 1

    @property
    def input_duration(self) -> Fraction:
        return Fraction(self.input_frames, 1) / self.input_fps

    @property
    def output_duration(self) -> Fraction:
        return Fraction(self.output_frames, 1) / self.output_fps

    def transitions(self) -> tuple[TransitionPlan, ...]:
        """모든 인접 원본 프레임 쌍의 생성 계획을 반환합니다."""

        cut_indices = set(self.cut_after)
        transitions: list[TransitionPlan] = []
        for left_index in range(self.input_frames - 1):
            if left_index in cut_indices:
                if self.scene_cut_policy is not SceneCutPolicy.HOLD_PREVIOUS:
                    raise ValueError(
                        f"unsupported scene-cut policy: {self.scene_cut_policy.value}"
                    )
                kind = TransitionKind.HOLD_PREVIOUS
            else:
                kind = TransitionKind.INTERPOLATE
            transitions.append(
                TransitionPlan(
                    left_index=left_index,
                    right_index=left_index + 1,
                    kind=kind,
                    generated_frames=self.multiplier - 1,
                )
            )
        return tuple(transitions)
