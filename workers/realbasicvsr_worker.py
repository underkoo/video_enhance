"""Pinned RealBasicVSR checkpoint를 실행하는 isolated NPY chunk worker입니다."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
from realbasicvsr_arch import RealBasicVSRNet

from rvfi_sr.realbasicvsr_contract import RealBasicVSRResolutionContract
from rvfi_sr.worker_protocol import WorkerRequest, WorkerResponse, WorkerStatus

_PROTOCOL_VERSION = 1
_BACKEND_ID = "mmagic-realbasicvsr"
_NATIVE_SCALE = 4
_CHECKPOINT_SHA256 = "52f77c2c835aaa3fe675b3959b2f85010a6c6f63f77f7e279394646e55a4e376"
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_ALLOWED_PARAMETERS = frozenset({"native_scale", "output_scale", "fp16", "gpu_index"})


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _load_input(path: Path) -> np.ndarray[Any, np.dtype[np.uint8]]:
    if path.suffix.casefold() != ".npy":
        raise ValueError("RealBasicVSR input_path must use the .npy suffix")
    frames = np.load(path, mmap_mode="r", allow_pickle=False)
    if frames.dtype != np.uint8:
        raise TypeError(f"input frames must be uint8, got {frames.dtype}")
    if frames.ndim != 4 or frames.shape[0] < 2 or frames.shape[-1] != 3:
        raise ValueError("input frames must have shape [N>=2,H,W,3]")
    if min(frames.shape[1:3]) < 64:
        raise ValueError("RealBasicVSR input height and width must be at least 64")
    return frames


def _validate_parameters(request: WorkerRequest) -> tuple[int, bool, int]:
    actual = frozenset(request.parameters)
    if actual != _ALLOWED_PARAMETERS:
        raise ValueError(
            f"RealBasicVSR parameters must be exactly {sorted(_ALLOWED_PARAMETERS)}"
        )
    native_scale = request.parameters["native_scale"]
    output_scale = request.parameters["output_scale"]
    fp16 = request.parameters["fp16"]
    gpu_index = request.parameters["gpu_index"]
    if native_scale != _NATIVE_SCALE:
        raise ValueError(f"native_scale must be {_NATIVE_SCALE}")
    if output_scale != 2:
        raise ValueError("only measured output_scale=2 is allowed")
    if not isinstance(fp16, bool):
        raise TypeError("fp16 must be boolean")
    if isinstance(gpu_index, bool) or not isinstance(gpu_index, int) or gpu_index < 0:
        raise ValueError("gpu_index must be a non-negative integer")
    return int(output_scale), fp16, gpu_index


def _load_model(checkpoint_path: Path, device: torch.device, fp16: bool) -> RealBasicVSRNet:
    checkpoint = checkpoint_path.resolve(strict=True)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual_sha256 = _sha256(checkpoint)
    if actual_sha256 != _CHECKPOINT_SHA256:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch: expected={_CHECKPOINT_SHA256}, "
            f"actual={actual_sha256}"
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint state_dict must be a mapping")
    prefix = "generator_ema."
    generator_state = {
        key.removeprefix(prefix): value
        for key, value in state_dict.items()
        if isinstance(key, str) and key.startswith(prefix)
    }
    if len(generator_state) != 320:
        raise RuntimeError(
            f"expected 320 generator_ema tensors, found {len(generator_state)}"
        )
    model = RealBasicVSRNet()
    model.load_state_dict(generator_state, strict=True)
    model.eval().requires_grad_(False).to(device)
    if fp16:
        model.half()
    return model


def _infer(
    frames: np.ndarray[Any, np.dtype[np.uint8]],
    model: RealBasicVSRNet,
    *,
    device: torch.device,
    fp16: bool,
    output_scale: int,
) -> np.ndarray[Any, np.dtype[np.uint8]]:
    tensor = torch.from_numpy(np.array(frames, dtype=np.uint8, order="C", copy=True))
    tensor = tensor.to(device=device, dtype=torch.float16 if fp16 else torch.float32)
    tensor = tensor.permute(0, 3, 1, 2).unsqueeze(0).div_(255.0)
    with torch.inference_mode():
        output = model(tensor, output_scale=output_scale)
    if output.shape != (
        1,
        frames.shape[0],
        3,
        frames.shape[1] * output_scale,
        frames.shape[2] * output_scale,
    ):
        raise RuntimeError(f"unexpected RealBasicVSR output shape: {tuple(output.shape)}")
    return (
        output[0]
        .float()
        .clamp_(0.0, 1.0)
        .mul_(255.0)
        .round_()
        .byte()
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )


def _write_atomic(output_path: Path, frames: np.ndarray[Any, Any]) -> Path:
    if output_path.suffix.casefold() != ".npy":
        raise ValueError("RealBasicVSR output_path must use the .npy suffix")
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")
    if partial.exists():
        raise FileExistsError(partial)
    try:
        with partial.open("xb") as stream:
            np.save(stream, frames, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, output_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return output_path


def _run(request: WorkerRequest, checkpoint: Path) -> WorkerResponse:
    if request.backend_id != _BACKEND_ID:
        raise ValueError(
            f"backend_id mismatch: expected={_BACKEND_ID!r}, actual={request.backend_id!r}"
        )
    output_scale, fp16, gpu_index = _validate_parameters(request)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the RealBasicVSR worker")
    if gpu_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"gpu_index out of range: index={gpu_index}, count={torch.cuda.device_count()}"
        )
    torch.cuda.set_device(gpu_index)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda", gpu_index)
    frames = _load_input(Path(request.input_path))
    RealBasicVSRResolutionContract.create(
        width=frames.shape[2],
        height=frames.shape[1],
        model_frames=frames.shape[0],
        output_scale=output_scale,
    )
    model = _load_model(checkpoint, device, fp16)
    torch.cuda.reset_peak_memory_stats(device)
    started_at = time.perf_counter()
    output_frames = _infer(
        frames,
        model,
        device=device,
        fp16=fp16,
        output_scale=output_scale,
    )
    elapsed_seconds = time.perf_counter() - started_at
    print(
        "RealBasicVSR inference metrics:",
        f"frames={frames.shape[0]}",
        f"source={frames.shape[2]}x{frames.shape[1]}",
        f"output_scale={output_scale}",
        f"elapsed_seconds={elapsed_seconds:.6f}",
        f"peak_allocated_bytes={torch.cuda.max_memory_allocated(device)}",
        f"peak_reserved_bytes={torch.cuda.max_memory_reserved(device)}",
        file=sys.stderr,
        flush=True,
    )
    output_path = _write_atomic(Path(request.output_path), output_frames)
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
    request = WorkerRequest.from_json(sys.stdin.read())
    try:
        response = _run(request, arguments.checkpoint)
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
