"""FlashVSR v1.1 Tiny 격리 환경에서 실행하는 `.npy` chunk worker입니다."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import sys
import traceback
from pathlib import Path
from typing import Any

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import torch.nn.functional as functional

from rvfi_sr.flashvsr_contract import FlashVSRChunkContract, FlashVSRResolutionContract
from rvfi_sr.worker_protocol import WorkerRequest, WorkerResponse, WorkerStatus

_PROTOCOL_VERSION = 1
_BACKEND_ID = "flashvsr-v1.1"
_NATIVE_SCALE = 4
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_PROMPT_SHA256 = "4601107a11e4e11a936a6b79df579e54dbc99872132bf542151f0ffd65b4b1ef"
_ALLOWED_PARAMETERS = frozenset(
    {
        "native_scale",
        "output_scale",
        "seed",
        "sparse_ratio",
        "kv_ratio",
        "local_range",
        "color_fix",
        "gpu_index",
    }
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


def _float_parameter(
    request: WorkerRequest,
    name: str,
    *,
    minimum: float,
) -> float:
    value = request.parameters.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"parameter {name!r} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result < minimum:
        raise ValueError(f"parameter {name!r} must be finite and >= {minimum}")
    return result


def _boolean_parameter(request: WorkerRequest, name: str) -> bool:
    value = request.parameters.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"parameter {name!r} must be boolean")
    return value


def _validate_request_parameters(
    request: WorkerRequest,
) -> tuple[int, int, int, float, float, int, bool, int]:
    actual = frozenset(request.parameters)
    unexpected = actual - _ALLOWED_PARAMETERS
    missing = _ALLOWED_PARAMETERS - actual
    if unexpected:
        raise ValueError(f"unexpected FlashVSR parameters: {sorted(unexpected)}")
    if missing:
        raise ValueError(f"missing FlashVSR parameters: {sorted(missing)}")
    native_scale = _integer_parameter(request, "native_scale", minimum=1)
    if native_scale != _NATIVE_SCALE:
        raise ValueError(f"native_scale must be {_NATIVE_SCALE}")
    output_scale = _integer_parameter(request, "output_scale", minimum=2)
    if output_scale not in {2, 4}:
        raise ValueError("output_scale must be one of 2, 4")
    local_range = _integer_parameter(request, "local_range", minimum=1)
    if local_range not in {9, 11}:
        raise ValueError("local_range must be one of 9, 11")
    return (
        native_scale,
        output_scale,
        _integer_parameter(request, "seed", minimum=0),
        _float_parameter(request, "sparse_ratio", minimum=0.1),
        _float_parameter(request, "kv_ratio", minimum=1.0),
        local_range,
        _boolean_parameter(request, "color_fix"),
        _integer_parameter(request, "gpu_index", minimum=0),
    )


def _load_input_frames(path: Path) -> np.ndarray[Any, np.dtype[np.uint8]]:
    if path.suffix.casefold() != ".npy":
        raise ValueError("FlashVSR input_path must use the .npy suffix")
    frames = np.load(path, mmap_mode="r", allow_pickle=False)
    if frames.dtype != np.uint8:
        raise TypeError(f"input frames must be uint8, got {frames.dtype}")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("input frames must have shape [N,H,W,3]")
    if frames.shape[1] < 1 or frames.shape[2] < 1:
        raise ValueError("input frame height and width must be positive")
    FlashVSRChunkContract.create(frames.shape[0])
    FlashVSRResolutionContract.create(width=frames.shape[2], height=frames.shape[1])
    return frames


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, *, sha256: str | None = None) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if sha256 is not None:
        actual_sha256 = _sha256(resolved)
        if actual_sha256 != sha256:
            raise RuntimeError(
                f"artifact SHA-256 mismatch: path={resolved}, "
                f"expected={sha256}, actual={actual_sha256}"
            )
    return resolved


def _load_pipeline(source_root: Path, checkpoint_root: Path, device: torch.device) -> Any:
    prompt_path = _require_file(
        source_root / "examples" / "WanVSR" / "prompt_tensor" / "posi_prompt.pth",
        sha256=_PROMPT_SHA256,
    )
    diffusion_path = _require_file(
        checkpoint_root / "diffusion_pytorch_model_streaming_dmd.safetensors"
    )
    projection_path = _require_file(checkpoint_root / "LQ_proj_in.ckpt")
    decoder_path = _require_file(checkpoint_root / "TCDecoder.ckpt")
    example_root = source_root / "examples" / "WanVSR"
    if not (source_root / "diffsynth" / "pipelines" / "flashvsr_tiny.py").is_file():
        raise FileNotFoundError(source_root / "diffsynth" / "pipelines" / "flashvsr_tiny.py")
    sys.path.insert(0, str(example_root))
    sys.path.insert(0, str(source_root))

    from diffsynth import FlashVSRTinyPipeline, ModelManager
    from utils.TCDecoder import build_tcdecoder
    from utils.utils import Causal_LQ4x_Proj

    manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    manager.load_models([str(diffusion_path)])
    pipeline = FlashVSRTinyPipeline.from_model_manager(
        manager,
        device=str(device),
    )
    pipeline.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(
        in_dim=3,
        out_dim=1536,
        layer_num=1,
    ).to(device=device, dtype=torch.bfloat16)
    projection_state = torch.load(projection_path, map_location="cpu", weights_only=True)
    pipeline.denoising_model().LQ_proj_in.load_state_dict(projection_state, strict=True)
    pipeline.denoising_model().LQ_proj_in.to(device)

    pipeline.TCDecoder = build_tcdecoder(
        new_channels=[512, 256, 128, 128],
        new_latent_channels=16 + 768,
    )
    decoder_state = torch.load(decoder_path, map_location="cpu", weights_only=True)
    pipeline.TCDecoder.load_state_dict(decoder_state, strict=True)

    pipeline.to(device)
    pipeline.enable_vram_management(num_persistent_param_in_dit=0)
    prompt_context = torch.load(prompt_path, map_location="cpu", weights_only=True)
    pipeline.init_cross_kv(context_tensor=prompt_context)
    pipeline.load_models_to_device([])
    return pipeline


def _prepare_input(
    frames: np.ndarray[Any, np.dtype[np.uint8]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, int, int]:
    frame_count, height, width, _ = frames.shape
    geometry = FlashVSRResolutionContract.create(width=width, height=height)
    padded_height = geometry.padded_height
    padded_width = geometry.padded_width
    source = torch.from_numpy(np.array(frames, dtype=np.uint8, order="C", copy=True)).to(
        device=device,
        dtype=torch.float32,
    )
    source = source.permute(0, 3, 1, 2).div_(255.0).mul_(2.0).sub_(1.0)
    source = functional.pad(
        source,
        (0, padded_width - width, 0, padded_height - height),
        mode="replicate",
    )
    source = functional.interpolate(
        source,
        scale_factor=float(_NATIVE_SCALE),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp_(-1.0, 1.0)
    source = torch.cat((source, source[-1:].expand(4, -1, -1, -1)), dim=0)
    expected_frames = FlashVSRChunkContract.create(frame_count).padded_frames
    if source.shape[0] != expected_frames:
        raise RuntimeError(
            f"FlashVSR terminal padding invariant failed: "
            f"expected={expected_frames}, actual={source.shape[0]}"
        )
    video = source.permute(1, 0, 2, 3).unsqueeze(0).to(torch.bfloat16)
    return video, padded_height * _NATIVE_SCALE, padded_width * _NATIVE_SCALE


def _infer(
    pipeline: Any,
    low_quality_video: torch.Tensor,
    *,
    source_frames: int,
    target_height: int,
    target_width: int,
    seed: int,
    sparse_ratio: float,
    kv_ratio: float,
    local_range: int,
    color_fix: bool,
) -> torch.Tensor:
    topk_ratio = sparse_ratio * 768 * 1280 / (target_height * target_width)
    result = pipeline(
        prompt="",
        negative_prompt="",
        cfg_scale=1.0,
        num_inference_steps=1,
        seed=seed,
        LQ_video=low_quality_video,
        num_frames=low_quality_video.shape[2],
        height=target_height,
        width=target_width,
        is_full_block=False,
        if_buffer=True,
        topk_ratio=topk_ratio,
        kv_ratio=kv_ratio,
        local_range=local_range,
        color_fix=False,
    )
    if result.ndim != 4 or result.shape[0] != 3:
        raise RuntimeError(f"unexpected FlashVSR tensor shape: {tuple(result.shape)}")
    if result.shape[1] != source_frames:
        raise RuntimeError(
            f"FlashVSR frame-count mismatch: expected={source_frames}, actual={result.shape[1]}"
        )
    if color_fix:
        corrected = pipeline.ColorCorrector(
            result.unsqueeze(0).to(device=low_quality_video.device),
            low_quality_video[:, :, : result.shape[1]],
            clip_range=(-1, 1),
            chunk_size=16,
            method="adain",
        )
        if corrected.shape != result.unsqueeze(0).shape:
            raise RuntimeError(
                f"FlashVSR color correction shape mismatch: {tuple(corrected.shape)}"
            )
        result = corrected[0]
    return result


def _to_uint8(
    result: torch.Tensor,
    *,
    height: int,
    width: int,
    output_scale: int,
) -> np.ndarray[Any, np.dtype[np.uint8]]:
    frames = result.permute(1, 0, 2, 3).float()
    if output_scale != _NATIVE_SCALE:
        frames = functional.interpolate(
            frames,
            scale_factor=output_scale / _NATIVE_SCALE,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    target_height = height * output_scale
    target_width = width * output_scale
    if frames.shape[2] < target_height or frames.shape[3] < target_width:
        raise RuntimeError(
            f"FlashVSR output is smaller than crop: tensor={tuple(frames.shape)}, "
            f"crop=({target_height}, {target_width})"
        )
    frames = frames[:, :, :target_height, :target_width]
    return (
        frames.add(1.0)
        .mul(127.5)
        .clamp(0.0, 255.0)
        .round()
        .byte()
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )


def _write_npy_atomic(output_path: Path, frames: np.ndarray[Any, Any]) -> Path:
    if output_path.suffix.casefold() != ".npy":
        raise ValueError("FlashVSR output_path must use the .npy suffix")
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


def _run(request: WorkerRequest, source_root: Path, checkpoint_root: Path) -> WorkerResponse:
    if request.backend_id != _BACKEND_ID:
        raise ValueError(
            f"backend_id mismatch: expected={_BACKEND_ID!r}, actual={request.backend_id!r}"
        )
    (
        _,
        output_scale,
        seed,
        sparse_ratio,
        kv_ratio,
        local_range,
        color_fix,
        gpu_index,
    ) = _validate_request_parameters(request)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the FlashVSR worker")
    if gpu_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"gpu_index out of range: index={gpu_index}, count={torch.cuda.device_count()}"
        )
    torch.cuda.set_device(gpu_index)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda", gpu_index)
    frames = _load_input_frames(Path(request.input_path))
    with contextlib.redirect_stdout(sys.stderr):
        pipeline = _load_pipeline(source_root, checkpoint_root, device)
        low_quality_video, target_height, target_width = _prepare_input(
            frames,
            device=device,
        )
        with torch.inference_mode():
            result = _infer(
                pipeline,
                low_quality_video,
                source_frames=frames.shape[0],
                target_height=target_height,
                target_width=target_width,
                seed=seed,
                sparse_ratio=sparse_ratio,
                kv_ratio=kv_ratio,
                local_range=local_range,
                color_fix=color_fix,
            )
            output_frames = _to_uint8(
                result,
                height=frames.shape[1],
                width=frames.shape[2],
                output_scale=output_scale,
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
