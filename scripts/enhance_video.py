#!/usr/bin/env python3
"""단일 MP4를 RIFE → RealBasicVSR → validated MP4로 처리합니다."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from rvfi_sr.cfr import probe_cfr_plan, stream_cfr_rgb24_frames
from rvfi_sr.chunk_io import (
    RealBasicVSRInputAssembler,
    RifeInputAssembler,
    stream_realbasicvsr_output_frames,
    stream_rife_output_frames,
)
from rvfi_sr.color import RgbDecodeContract
from rvfi_sr.config import PipelineConfig
from rvfi_sr.config_loader import load_hydra_config
from rvfi_sr.encode import AtomicMp4Encoder, EncodeContract
from rvfi_sr.media_tools import run_ffprobe
from rvfi_sr.model_specs import MODEL_PROVENANCE
from rvfi_sr.provenance import write_provenance_manifest
from rvfi_sr.realbasicvsr_contract import (
    RealBasicVSRResolutionContract,
    plan_realbasicvsr_chunks,
)
from rvfi_sr.rife_chunks import plan_rife_chunks
from rvfi_sr.scene_cut import detect_scene_cuts
from rvfi_sr.timeline import TimelineContract
from rvfi_sr.worker_protocol import WorkerRequest, WorkerResponse
from rvfi_sr.worker_runner import PersistentWorker, run_worker


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", default="deterministic")
    parser.add_argument("--keep-work", action="store_true")
    return parser.parse_args()


def _load_config(
    repo_root: Path,
    *,
    config_name: str,
    input_path: Path,
    output_path: Path,
) -> PipelineConfig:
    previous_input = os.environ.get("VIDEO_ENHANCE_INPUT")
    previous_output = os.environ.get("VIDEO_ENHANCE_OUTPUT")
    os.environ["VIDEO_ENHANCE_INPUT"] = str(input_path.parent)
    os.environ["VIDEO_ENHANCE_OUTPUT"] = str(output_path.parent)
    try:
        return load_hydra_config(repo_root / "configs", config_name)
    finally:
        if previous_input is None:
            os.environ.pop("VIDEO_ENHANCE_INPUT", None)
        else:
            os.environ["VIDEO_ENHANCE_INPUT"] = previous_input
        if previous_output is None:
            os.environ.pop("VIDEO_ENHANCE_OUTPUT", None)
        else:
            os.environ["VIDEO_ENHANCE_OUTPUT"] = previous_output


def _validated_paths(arguments: argparse.Namespace, repo_root: Path) -> tuple[Path, Path]:
    input_path = arguments.input.resolve(strict=True)
    output_path = arguments.output.resolve(strict=False)
    if not input_path.is_file() or input_path.suffix.casefold() != ".mp4":
        raise ValueError("--input must be an existing MP4 file")
    if not output_path.is_absolute() or output_path.suffix.casefold() != ".mp4":
        raise ValueError("--output must be an absolute MP4 path")
    if input_path == output_path:
        raise ValueError("input and output paths must differ")
    provenance_path = output_path.with_name(f"{output_path.stem}.provenance.json")
    occupied = tuple(
        path
        for path in (
            output_path,
            output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}"),
            provenance_path,
            provenance_path.with_name(
                f"{provenance_path.stem}.partial{provenance_path.suffix}"
            ),
        )
        if path.exists()
    )
    if occupied:
        raise FileExistsError(occupied[0])
    if not repo_root.is_dir():
        raise NotADirectoryError(repo_root)
    return input_path, output_path


def _runtime_paths(repo_root: Path) -> dict[str, Path]:
    paths = {
        "ffmpeg": (
            repo_root
            / ".runtime/tools/ffmpeg-n8.1.2-34-g9b6c8969e0/bin/ffmpeg"
        ),
        "ffprobe": (
            repo_root
            / ".runtime/tools/ffmpeg-n8.1.2-34-g9b6c8969e0/bin/ffprobe"
        ),
        "rife_python": repo_root / ".runtime/envs/practical-rife-v4.25/bin/python",
        "rife_source": repo_root / ".runtime/sources/practical-rife-v4.25",
        "rife_checkpoint": repo_root / ".runtime/checkpoints/practical-rife-v4.25",
        "vsr_python": repo_root / ".runtime/envs/mmagic-realbasicvsr/bin/python",
        "vsr_checkpoint": (
            repo_root
            / ".runtime/checkpoints/mmagic-realbasicvsr/RealBasicVSR.pth"
        ),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"runtime component is missing: {name}={path}")
    return paths


def _validate_worker_shape(
    response: WorkerResponse,
    *,
    frames: int,
    width: int,
    height: int,
) -> None:
    actual = (response.frame_count, response.width, response.height, response.dtype)
    expected = (frames, width, height, "uint8")
    if actual != expected:
        raise RuntimeError(
            f"worker output contract mismatch: expected={expected}, actual={actual}"
        )


def _run_pipeline(
    repo_root: Path,
    input_path: Path,
    output_path: Path,
    config: PipelineConfig,
    runtime: dict[str, Path],
    work_dir: Path,
) -> Path:
    if config.vfi.backend_id != "practical-rife-v4.25":
        raise ValueError("this entrypoint only supports practical-rife-v4.25")
    if config.vsr.backend_id != "mmagic-realbasicvsr":
        raise ValueError("this entrypoint only supports mmagic-realbasicvsr")
    if config.order.value != "vfi_then_vsr":
        raise ValueError("this entrypoint only supports vfi_then_vsr")
    if output_path.parent != config.output_dir:
        raise ValueError(f"output parent must equal configured output_dir: {config.output_dir}")
    if config.runtime.final_output_root is None:
        raise ValueError("runtime.final_output_root must be configured")

    spec = run_ffprobe(runtime["ffprobe"], input_path)
    if spec.audio is not None and spec.audio.codec != "aac":
        raise ValueError(f"only AAC audio remux is supported, got {spec.audio.codec!r}")
    cfr_plan = probe_cfr_plan(
        runtime["ffmpeg"],
        input_path,
        expected_source_frames=spec.frame_count,
        target_fps=config.cfr.target_fps,
    )
    scene = detect_scene_cuts(
        runtime["ffmpeg"],
        input_path,
        expected_frames=cfr_plan.output_frames,
        target_fps=config.cfr.target_fps,
        threshold=config.scene_cut.threshold,
    )
    color = RgbDecodeContract.create(
        spec.color,
        untagged_range=config.color.untagged_range,
        untagged_space=config.color.untagged_space,
    )
    timeline = TimelineContract.create(
        cfr_plan.output_frames,
        config.cfr.target_fps,
        config.vfi.temporal_multiplier,
        scene.cut_after,
    )
    print(
        f"preflight: source={spec.width}x{spec.height}/{spec.frame_count} frames, "
        f"CFR={cfr_plan.output_frames}, cuts={len(scene.cut_after)}, "
        f"output={spec.width * 2}x{spec.height * 2}/{timeline.output_frames} frames",
        flush=True,
    )

    rife_chunks = plan_rife_chunks(
        frame_count=cfr_plan.output_frames,
        cut_after=scene.cut_after,
        multiplier=config.vfi.temporal_multiplier,
    )
    rife_input_dir = work_dir / "rife-input"
    rife_output_dir = work_dir / "rife-output"
    rife_assembler = RifeInputAssembler(
        rife_chunks,
        width=spec.width,
        height=spec.height,
        output_dir=rife_input_dir,
    )
    stream_cfr_rgb24_frames(
        runtime["ffmpeg"],
        input_path,
        width=spec.width,
        height=spec.height,
        plan=cfr_plan,
        color=color,
        consume_frame=rife_assembler.consume,
    )
    rife_input_paths = rife_assembler.finalize()
    rife_output_dir.mkdir(parents=True, exist_ok=False)
    rife_output_paths: list[Path | None] = []
    worker_environment = {"PYTHONPATH": str(repo_root / "src")}
    for index, (chunk, chunk_input) in enumerate(
        zip(rife_chunks, rife_input_paths, strict=True)
    ):
        if not chunk.use_model:
            rife_output_paths.append(None)
            continue
        chunk_output = rife_output_dir / f"rife-output-{index:06d}.npy"
        request = WorkerRequest.create(
            job_id=f"rife-{index:06d}",
            backend_id=config.vfi.backend_id,
            input_path=chunk_input,
            output_path=chunk_output,
            parameters={
                "temporal_multiplier": config.vfi.temporal_multiplier,
                "fp16": config.runtime.fp16,
                "inference_scale": 1.0,
                "gpu_index": config.runtime.gpu_index,
            },
        )
        print(f"RIFE chunk {index + 1}/{len(rife_chunks)}", flush=True)
        response = run_worker(
            (
                str(runtime["rife_python"]),
                str(repo_root / "workers/practical_rife_worker.py"),
                "--source-root",
                str(runtime["rife_source"]),
                "--checkpoint-root",
                str(runtime["rife_checkpoint"]),
            ),
            request,
            timeout_seconds=config.runtime.worker_timeout_seconds,
            environment=worker_environment,
        )
        _validate_worker_shape(
            response,
            frames=chunk.worker_output_frames,
            width=spec.width,
            height=spec.height,
        )
        rife_output_paths.append(chunk_output)

    max_vsr_frames = RealBasicVSRResolutionContract.max_model_frames(
        width=spec.width,
        height=spec.height,
    )
    vsr_chunks = plan_realbasicvsr_chunks(
        frame_count=timeline.output_frames,
        cut_after=timeline.output_cut_after,
        max_source_frames=max_vsr_frames,
    )
    vsr_input_dir = work_dir / "vsr-input"
    vsr_assembler = RealBasicVSRInputAssembler(
        vsr_chunks,
        width=spec.width,
        height=spec.height,
        output_dir=vsr_input_dir,
    )
    stream_rife_output_frames(
        rife_chunks,
        rife_input_paths,
        tuple(rife_output_paths),
        width=spec.width,
        height=spec.height,
        consume_frame=vsr_assembler.consume,
    )
    vsr_input_paths = vsr_assembler.finalize()
    shutil.rmtree(rife_input_dir)
    shutil.rmtree(rife_output_dir)

    vsr_output_dir = work_dir / "vsr-output"
    vsr_output_dir.mkdir(parents=True, exist_ok=False)
    vsr_output_paths: list[Path] = []
    with PersistentWorker(
        (
            str(runtime["vsr_python"]),
            str(repo_root / "workers/realbasicvsr_worker.py"),
            "--checkpoint",
            str(runtime["vsr_checkpoint"]),
            "--persistent",
        ),
        timeout_seconds=config.runtime.worker_timeout_seconds,
        environment=worker_environment,
    ) as vsr_worker:
        for index, (chunk, chunk_input) in enumerate(
            zip(vsr_chunks, vsr_input_paths, strict=True)
        ):
            chunk_output = vsr_output_dir / f"vsr-output-{index:06d}.npy"
            request = WorkerRequest.create(
                job_id=f"realbasicvsr-{index:06d}",
                backend_id=config.vsr.backend_id,
                input_path=chunk_input,
                output_path=chunk_output,
                parameters={
                    "native_scale": 4,
                    "output_scale": config.vsr.spatial_scale,
                    "fp16": config.runtime.fp16,
                    "gpu_index": config.runtime.gpu_index,
                },
            )
            print(f"RealBasicVSR chunk {index + 1}/{len(vsr_chunks)}", flush=True)
            response = vsr_worker.run(request)
            _validate_worker_shape(
                response,
                frames=chunk.model_frames,
                width=spec.width * config.vsr.spatial_scale,
                height=spec.height * config.vsr.spatial_scale,
            )
            vsr_output_paths.append(chunk_output)

    encode_contract = EncodeContract.create(
        input_path=input_path,
        output_path=output_path,
        final_output_root=config.runtime.final_output_root,
        width=spec.width * config.vsr.spatial_scale,
        height=spec.height * config.vsr.spatial_scale,
        fps=timeline.output_fps,
        frame_count=timeline.output_frames,
        expect_audio=spec.audio is not None,
        source_audio_duration=(spec.audio.duration if spec.audio is not None else None),
        crf=config.encode.crf,
        preset=config.encode.preset,
    )
    encoder = AtomicMp4Encoder(
        encode_contract,
        ffmpeg_path=runtime["ffmpeg"],
        timeout_seconds=config.runtime.worker_timeout_seconds,
    )
    finalized = False
    try:
        stream_realbasicvsr_output_frames(
            vsr_chunks,
            tuple(vsr_output_paths),
            width=spec.width,
            height=spec.height,
            output_scale=config.vsr.spatial_scale,
            consume_frame=encoder.consume,
        )
        result = encoder.finalize(runtime["ffprobe"])
        finalized = True
        provenance_path = output_path.with_name(
            f"{output_path.stem}.provenance.json"
        )
        write_provenance_manifest(
            provenance_path,
            (
                MODEL_PROVENANCE[config.vfi.backend_id],
                MODEL_PROVENANCE[config.vsr.backend_id],
            ),
        )
        return result
    finally:
        if not finalized:
            encoder.abort()


def main() -> int:
    arguments = _arguments()
    repo_root = Path(__file__).resolve().parents[1]
    input_path, output_path = _validated_paths(arguments, repo_root)
    config = _load_config(
        repo_root,
        config_name=arguments.config,
        input_path=input_path,
        output_path=output_path,
    )
    runtime = _runtime_paths(repo_root)
    jobs_root = repo_root / ".runtime/jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="enhance-", dir=jobs_root))
    succeeded = False
    try:
        result = _run_pipeline(
            repo_root,
            input_path,
            output_path,
            config,
            runtime,
            work_dir,
        )
        succeeded = True
        print(f"completed: {result}", flush=True)
        return 0
    finally:
        if succeeded and not arguments.keep_work:
            shutil.rmtree(work_dir)
        else:
            print(f"work directory retained: {work_dir}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
