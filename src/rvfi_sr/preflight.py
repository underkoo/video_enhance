"""실제 추론 전에 GPU 하드웨어 계약을 검사합니다."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

_COMPUTE_PATTERN = re.compile(r"(\d+)\.(\d+)")


@dataclass(frozen=True, slots=True)
class GpuSpec:
    """locale과 단위가 제거된 NVIDIA GPU 명세입니다."""

    index: int
    name: str
    memory_total_mib: int
    compute_capability: tuple[int, int]

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or self.index < 0:
            raise ValueError("GPU index must be a non-negative integer")
        if not self.name.strip():
            raise ValueError("GPU name must not be empty")
        if isinstance(self.memory_total_mib, bool) or self.memory_total_mib < 1:
            raise ValueError("GPU memory must be positive")
        if len(self.compute_capability) != 2 or any(
            isinstance(value, bool) or value < 0 for value in self.compute_capability
        ):
            raise ValueError("compute capability must contain non-negative major/minor")

    @classmethod
    def from_nvidia_smi_csv(cls, payload: str) -> GpuSpec:
        """`csv,noheader,nounits` 한 행을 엄격하게 파싱합니다."""

        rows = [line.strip() for line in payload.splitlines() if line.strip()]
        if len(rows) != 1:
            raise ValueError(f"expected exactly one GPU row, found {len(rows)}")
        fields = [field.strip() for field in rows[0].split(",")]
        if len(fields) != 4:
            raise ValueError(f"expected exactly four GPU fields, found {len(fields)}")
        compute_match = _COMPUTE_PATTERN.fullmatch(fields[3])
        if compute_match is None:
            raise ValueError("compute capability must use major.minor format")
        try:
            index = int(fields[0])
            memory_total_mib = int(fields[2])
        except ValueError as error:
            raise ValueError("GPU index and memory must be integers") from error
        return cls(
            index=index,
            name=fields[1],
            memory_total_mib=memory_total_mib,
            compute_capability=(
                int(compute_match.group(1)),
                int(compute_match.group(2)),
            ),
        )


def query_gpu(gpu_index: int) -> GpuSpec:
    """nvidia-smi로 선택한 GPU 하나만 조회합니다."""

    if isinstance(gpu_index, bool) or gpu_index < 0:
        raise ValueError("gpu_index must be a non-negative integer")
    command = (
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=index,name,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    )
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        raise RuntimeError("nvidia-smi GPU query failed") from error
    return GpuSpec.from_nvidia_smi_csv(result.stdout)


def validate_gpu(
    gpu: GpuSpec,
    *,
    expected_index: int,
    minimum_memory_mib: int,
    minimum_compute: tuple[int, int],
) -> None:
    """backend 환경 생성 전에 GPU index, VRAM, compute capability를 검사합니다."""

    if gpu.index != expected_index:
        raise RuntimeError(
            f"selected GPU index mismatch: expected={expected_index}, actual={gpu.index}"
        )
    if gpu.memory_total_mib < minimum_memory_mib:
        raise RuntimeError(
            f"insufficient VRAM: required={minimum_memory_mib} MiB, "
            f"actual={gpu.memory_total_mib} MiB"
        )
    if gpu.compute_capability < minimum_compute:
        required = ".".join(map(str, minimum_compute))
        actual = ".".join(map(str, gpu.compute_capability))
        raise RuntimeError(
            f"insufficient compute capability: required={required}, actual={actual}"
        )
