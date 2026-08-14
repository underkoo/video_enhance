"""Practical-RIFE v4.25 격리 환경에서 실행하는 `.npy` chunk worker입니다."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import sys
import traceback
from pathlib import Path
from typing import Any

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import torch.nn.functional as functional

from rvfi_sr.worker_protocol import WorkerRequest, WorkerResponse, WorkerStatus

_PROTOCOL_VERSION = 1
_BACKEND_ID = "practical-rife-v4.25"
_ALIGNMENT = 128
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_ALLOWED_PARAMETERS = frozenset(
    {"temporal_multiplier", "fp16", "inference_scale", "gpu_index"}
)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    return parser.parse_args()


def _absolute_directory(path: Path, name: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    return resolved


def _integer_parameter(request: WorkerRequest, name: str, *, minimum: int) -> int:
    value = request.parameters.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"parameter {name!r} must be an integer >= {minimum}")
    return value


def _boolean_parameter(request: WorkerRequest, name: str) -> bool:
    value = request.parameters.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"parameter {name!r} must be boolean")
    return value


def _float_parameter(request: WorkerRequest, name: str) -> float:
    value = request.parameters.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"parameter {name!r} must be numeric")
    result = float(value)
    if result not in {0.25, 0.5, 1.0, 2.0, 4.0}:
        raise ValueError("inference_scale must be one of 0.25, 0.5, 1.0, 2.0, 4.0")
    return result


def _validate_request_parameters(request: WorkerRequest) -> tuple[int, bool, float, int]:
    unexpected = frozenset(request.parameters) - _ALLOWED_PARAMETERS
    missing = _ALLOWED_PARAMETERS - frozenset(request.parameters)
    if unexpected:
        raise ValueError(f"unexpected RIFE parameters: {sorted(unexpected)}")
    if missing:
        raise ValueError(f"missing RIFE parameters: {sorted(missing)}")
    multiplier = _integer_parameter(request, "temporal_multiplier", minimum=2)
    if multiplier not in {2, 4, 8}:
        raise ValueError("temporal_multiplier must be one of 2, 4, 8")
    return (
        multiplier,
        _boolean_parameter(request, "fp16"),
        _float_parameter(request, "inference_scale"),
        _integer_parameter(request, "gpu_index", minimum=0),
    )


def _load_input_frames(path: Path) -> np.ndarray[Any, np.dtype[np.uint8]]:
    if path.suffix.casefold() != ".npy":
        raise ValueError("RIFE input_path must use the .npy suffix")
    frames = np.load(path, mmap_mode="r", allow_pickle=False)
    if frames.dtype != np.uint8:
        raise TypeError(f"input frames must be uint8, got {frames.dtype}")
    if frames.ndim != 4 or frames.shape[0] < 2 or frames.shape[-1] != 3:
        raise ValueError("input frames must have shape [N>=2,H,W,3]")
    if frames.shape[1] < 1 or frames.shape[2] < 1:
        raise ValueError("input frame height and width must be positive")
    return frames


def _load_model(source_root: Path, checkpoint_root: Path, fp16: bool) -> Any:
    if not (source_root / "model" / "warplayer.py").is_file():
        raise FileNotFoundError(source_root / "model" / "warplayer.py")
    if not (checkpoint_root / "train_log" / "RIFE_HDv3.py").is_file():
        raise FileNotFoundError(checkpoint_root / "train_log" / "RIFE_HDv3.py")
    if not (checkpoint_root / "train_log" / "flownet.pkl").is_file():
        raise FileNotFoundError(checkpoint_root / "train_log" / "flownet.pkl")

    sys.path.insert(0, str(checkpoint_root))
    sys.path.insert(0, str(source_root))
    # Official v4.25 FP16 code relies on the default tensor dtype when warplayer
    # creates its cached sampling grid. Weight/input casting alone leaves that
    # grid as float32 and makes grid_sample fail with a dtype mismatch.
    torch.set_default_dtype(torch.float16 if fp16 else torch.float32)
    module = importlib.import_module("train_log.RIFE_HDv3")
    model = module.Model()
    if getattr(model, "version", None) != 4.25:
        raise RuntimeError(f"unexpected RIFE model version: {getattr(model, 'version', None)!r}")
    model.load_model(str(checkpoint_root / "train_log"), -1)
    model.eval()
    model.device()
    return model


def _to_tensor(
    frame: np.ndarray[Any, np.dtype[np.uint8]],
    *,
    device: torch.device,
    fp16: bool,
    padded_height: int,
    padded_width: int,
) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1)))
    tensor = tensor.to(device=device, dtype=torch.float32, non_blocking=False).unsqueeze(0)
    tensor = tensor / 255.0
    tensor = functional.pad(
        tensor,
        (0, padded_width - frame.shape[1], 0, padded_height - frame.shape[0]),
    )
    return tensor.half() if fp16 else tensor


def _to_uint8_frame(tensor: torch.Tensor, height: int, width: int) -> np.ndarray[Any, Any]:
    return (
        tensor[0, :, :height, :width]
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def _interpolate_chunk(
    frames: np.ndarray[Any, np.dtype[np.uint8]],
    model: Any,
    *,
    multiplier: int,
    fp16: bool,
    inference_scale: float,
    device: torch.device,
) -> np.ndarray[Any, np.dtype[np.uint8]]:
    frame_count, height, width, channels = frames.shape
    padded_height = ((height + _ALIGNMENT - 1) // _ALIGNMENT) * _ALIGNMENT
    padded_width = ((width + _ALIGNMENT - 1) // _ALIGNMENT) * _ALIGNMENT
    output = np.empty((frame_count * multiplier, height, width, channels), dtype=np.uint8)
    output_index = 0

    with torch.inference_mode():
        previous = _to_tensor(
            frames[0],
            device=device,
            fp16=fp16,
            padded_height=padded_height,
            padded_width=padded_width,
        )
        for frame_index in range(frame_count - 1):
            current = _to_tensor(
                frames[frame_index + 1],
                device=device,
                fp16=fp16,
                padded_height=padded_height,
                padded_width=padded_width,
            )
            output[output_index] = frames[frame_index]
            output_index += 1
            for timestep_index in range(1, multiplier):
                timestep = timestep_index / multiplier
                interpolated = model.inference(
                    previous,
                    current,
                    timestep=timestep,
                    scale=inference_scale,
                )
                output[output_index] = _to_uint8_frame(interpolated, height, width)
                output_index += 1
            previous = current

    output[output_index] = frames[-1]
    output_index += 1
    terminal_holds = multiplier - 1
    output[output_index : output_index + terminal_holds] = frames[-1]
    output_index += terminal_holds
    if output_index != output.shape[0]:
        raise RuntimeError(
            f"RIFE output frame-count invariant failed: expected={output.shape[0]}, "
            f"actual={output_index}"
        )
    return output


def _write_npy_atomic(output_path: Path, frames: np.ndarray[Any, Any]) -> Path:
    if output_path.suffix.casefold() != ".npy":
        raise ValueError("RIFE output_path must use the .npy suffix")
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")
    if partial_path.exists():
        raise FileExistsError(partial_path)
    try:
        with partial_path.open("xb") as stream:
            np.save(stream, frames, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial_path, output_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise
    return output_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _run(request: WorkerRequest, source_root: Path, checkpoint_root: Path) -> WorkerResponse:
    if request.backend_id != _BACKEND_ID:
        raise ValueError(
            f"backend_id mismatch: expected={_BACKEND_ID!r}, actual={request.backend_id!r}"
        )
    multiplier, fp16, inference_scale, gpu_index = _validate_request_parameters(request)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Practical-RIFE worker")
    if gpu_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"gpu_index out of range: index={gpu_index}, count={torch.cuda.device_count()}"
        )
    torch.cuda.set_device(gpu_index)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda", gpu_index)
    frames = _load_input_frames(Path(request.input_path))
    model = _load_model(source_root, checkpoint_root, fp16)
    output_frames = _interpolate_chunk(
        frames,
        model,
        multiplier=multiplier,
        fp16=fp16,
        inference_scale=inference_scale,
        device=device,
    )
    output_path = _write_npy_atomic(Path(request.output_path), output_frames)
    return WorkerResponse(
        schema_version=_PROTOCOL_VERSION,
        job_id=request.job_id,
        status=WorkerStatus.SUCCEEDED,
        output_sha256=_sha256(output_path),
        frame_count=output_frames.shape[0],
        width=output_frames.shape[2],
        height=output_frames.shape[1],
        dtype=str(output_frames.dtype),
    )


def main() -> int:
    """stdin 요청 하나를 처리하고 stdout에 terminal 응답 하나만 씁니다."""

    arguments = _parse_arguments()
    source_root = _absolute_directory(arguments.source_root, "source_root")
    checkpoint_root = _absolute_directory(arguments.checkpoint_root, "checkpoint_root")
    request = WorkerRequest.from_json(sys.stdin.read())
    try:
        response = _run(request, source_root, checkpoint_root)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        response = WorkerResponse(
            schema_version=_PROTOCOL_VERSION,
            job_id=request.job_id,
            status=WorkerStatus.FAILED,
            output_sha256=None,
            frame_count=None,
            width=None,
            height=None,
            dtype=None,
            error_type=type(error).__name__,
            error_message=str(error),
        )
        print(response.to_json(), flush=True)
        return 1
    print(response.to_json(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
