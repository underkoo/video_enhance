#!/usr/bin/env python3
"""단일 MP4를 RIFE → RealBasicVSR → validated MP4로 처리합니다."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from fractions import Fraction
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
from rvfi_sr.receipts import (
    fingerprint_text,
    receipt_path,
    sha256_file,
    validate_npy_receipt,
    write_npy_receipt,
)
from rvfi_sr.rife_chunks import plan_rife_chunks
from rvfi_sr.run_manifest import (
    RunPlanManifest,
    validate_run_plan,
    write_completed_run_manifest,
    write_run_plan,
)
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
    parser.add_argument("--resume-work", type=Path)
    parser.add_argument("--checkpoint-after-vsr-chunks", type=int)
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
    run_manifest_path = output_path.with_name(f"{output_path.stem}.run.json")
    occupied = tuple(
        path
        for path in (
            output_path,
            output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}"),
            provenance_path,
            provenance_path.with_name(
                f"{provenance_path.stem}.partial{provenance_path.suffix}"
            ),
            run_manifest_path,
            run_manifest_path.with_name(
                f"{run_manifest_path.stem}.partial{run_manifest_path.suffix}"
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


def _fraction_string(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _artifact_fingerprint(
    plan: RunPlanManifest,
    *,
    stage: str,
    index: int,
    chunk: object,
) -> str:
    payload = {
        "plan": json.loads(plan.to_json()),
        "stage": stage,
        "index": index,
        "chunk": asdict(chunk),
    }
    return fingerprint_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _validate_or_missing_artifact(
    path: Path,
    *,
    producer_sha256: str,
    expected_shape: tuple[int, int, int, int],
) -> bool:
    artifact_exists = path.is_file()
    sidecar_exists = receipt_path(path).is_file()
    if artifact_exists != sidecar_exists:
        raise RuntimeError(
            f"artifact/receipt completeness mismatch: artifact={path}, "
            f"receipt={receipt_path(path)}"
        )
    if not artifact_exists:
        return False
    validate_npy_receipt(
        path,
        producer_sha256=producer_sha256,
        expected_shape=expected_shape,
    )
    return True


def _validate_complete_stage(
    paths: tuple[Path, ...],
    producers: tuple[str, ...],
    shapes: tuple[tuple[int, int, int, int], ...],
) -> bool:
    if not (len(paths) == len(producers) == len(shapes)):
        raise ValueError("stage artifact metadata counts must match")
    states = tuple(
        _validate_or_missing_artifact(
            path,
            producer_sha256=producer,
            expected_shape=shape,
        )
        for path, producer, shape in zip(paths, producers, shapes, strict=True)
    )
    if any(states) and not all(states):
        raise RuntimeError("input assembly stage is incomplete; partial reuse is forbidden")
    return all(states)


def _write_stage_receipts(paths: tuple[Path, ...], producers: tuple[str, ...]) -> None:
    if len(paths) != len(producers):
        raise ValueError("artifact and producer counts must match")
    for path, producer in zip(paths, producers, strict=True):
        write_npy_receipt(path, producer_sha256=producer)


def _run_forward_pipeline(
    repo_root: Path,
    input_path: Path,
    output_path: Path,
    config: PipelineConfig,
    runtime: dict[str, Path],
    work_dir: Path,
    *,
    resume: bool,
    checkpoint_after_vsr_chunks: int | None,
) -> Path:
    started_at = time.perf_counter()
    if config.vfi.backend_id != "practical-rife-v4.25":
        raise ValueError("this entrypoint only supports practical-rife-v4.25")
    if config.vsr.backend_id != "mmagic-realbasicvsr":
        raise ValueError("this entrypoint only supports mmagic-realbasicvsr")
    if config.order.value != "vfi_then_vsr":
        raise ValueError("forward pipeline requires vfi_then_vsr")
    if output_path.parent != config.output_dir:
        raise ValueError(f"output parent must equal configured output_dir: {config.output_dir}")
    if config.runtime.final_output_root is None:
        raise ValueError("runtime.final_output_root must be configured")
    if checkpoint_after_vsr_chunks is not None and checkpoint_after_vsr_chunks < 1:
        raise ValueError("checkpoint_after_vsr_chunks must be positive")

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
    max_vsr_frames = RealBasicVSRResolutionContract.max_model_frames(
        width=spec.width,
        height=spec.height,
    )
    vsr_chunks = plan_realbasicvsr_chunks(
        frame_count=timeline.output_frames,
        cut_after=timeline.output_cut_after,
        max_source_frames=max_vsr_frames,
    )
    run_plan = RunPlanManifest(
        schema_version=2,
        input_path=str(input_path),
        output_path=str(output_path),
        input_sha256=sha256_file(input_path),
        config_sha256=fingerprint_text(config.model_dump_json()),
        order=config.order.value,
        source_width=spec.width,
        source_height=spec.height,
        source_frames=spec.frame_count,
        cfr_frames=cfr_plan.output_frames,
        cfr_dropped_frames=cfr_plan.dropped_frames,
        cfr_duplicated_frames=cfr_plan.duplicated_frames,
        target_fps=_fraction_string(config.cfr.target_fps),
        scene_cut_after=scene.cut_after,
        vfi_backend_id=config.vfi.backend_id,
        vfi_multiplier=config.vfi.temporal_multiplier,
        rife_chunk_count=len(rife_chunks),
        vsr_backend_id=config.vsr.backend_id,
        vsr_scale=config.vsr.spatial_scale,
        vsr_max_model_frames=max_vsr_frames,
        vsr_chunk_count=len(vsr_chunks),
        output_fps=_fraction_string(timeline.output_fps),
        output_frames=timeline.output_frames,
    )
    work_manifest_path = work_dir / "run-plan.json"
    if resume:
        validate_run_plan(work_manifest_path, run_plan)
        print(f"resume plan validated: {work_dir}", flush=True)
    else:
        write_run_plan(work_manifest_path, run_plan)

    rife_input_dir = work_dir / "rife-input"
    rife_output_dir = work_dir / "rife-output"
    rife_input_paths = tuple(
        rife_input_dir / f"rife-input-{index:06d}.npy"
        for index in range(len(rife_chunks))
    )
    rife_input_producers = tuple(
        _artifact_fingerprint(
            run_plan,
            stage="rife-input",
            index=index,
            chunk=chunk,
        )
        for index, chunk in enumerate(rife_chunks)
    )
    rife_input_shapes = tuple(
        (chunk.source_frames, spec.height, spec.width, 3) for chunk in rife_chunks
    )
    vsr_input_dir = work_dir / "vsr-input"
    vsr_input_paths = tuple(
        vsr_input_dir / f"realbasicvsr-input-{index:06d}.npy"
        for index in range(len(vsr_chunks))
    )
    vsr_input_producers = tuple(
        _artifact_fingerprint(
            run_plan,
            stage="realbasicvsr-input",
            index=index,
            chunk=chunk,
        )
        for index, chunk in enumerate(vsr_chunks)
    )
    vsr_input_shapes = tuple(
        (chunk.model_frames, spec.height, spec.width, 3) for chunk in vsr_chunks
    )
    vsr_inputs_ready = _validate_complete_stage(
        vsr_input_paths,
        vsr_input_producers,
        vsr_input_shapes,
    )
    worker_environment = {"PYTHONPATH": str(repo_root / "src")}
    if vsr_inputs_ready:
        print(f"reused {len(vsr_input_paths)} validated RealBasicVSR inputs", flush=True)
    else:
        rife_inputs_ready = _validate_complete_stage(
            rife_input_paths,
            rife_input_producers,
            rife_input_shapes,
        )
        if rife_inputs_ready:
            print(f"reused {len(rife_input_paths)} validated RIFE inputs", flush=True)
        else:
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
            assembled_rife_inputs = rife_assembler.finalize()
            if assembled_rife_inputs != rife_input_paths:
                raise RuntimeError("RIFE input assembler returned unexpected paths")
            _write_stage_receipts(rife_input_paths, rife_input_producers)

        rife_output_dir.mkdir(parents=True, exist_ok=True)
        rife_output_paths: list[Path | None] = []
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
            producer = fingerprint_text(request.to_json())
            expected_shape = (
                chunk.worker_output_frames,
                spec.height,
                spec.width,
                3,
            )
            if _validate_or_missing_artifact(
                chunk_output,
                producer_sha256=producer,
                expected_shape=expected_shape,
            ):
                print(f"reused RIFE chunk {index + 1}/{len(rife_chunks)}", flush=True)
            else:
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
                write_npy_receipt(chunk_output, producer_sha256=producer)
            rife_output_paths.append(chunk_output)

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
        assembled_vsr_inputs = vsr_assembler.finalize()
        if assembled_vsr_inputs != vsr_input_paths:
            raise RuntimeError("RealBasicVSR input assembler returned unexpected paths")
        _write_stage_receipts(vsr_input_paths, vsr_input_producers)
        shutil.rmtree(rife_input_dir)
        shutil.rmtree(rife_output_dir)

    vsr_output_dir = work_dir / "vsr-output"
    vsr_output_dir.mkdir(parents=True, exist_ok=True)
    vsr_output_paths: list[Path] = []
    new_vsr_chunks = 0
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
            producer = fingerprint_text(request.to_json())
            expected_shape = (
                chunk.model_frames,
                spec.height * config.vsr.spatial_scale,
                spec.width * config.vsr.spatial_scale,
                3,
            )
            if _validate_or_missing_artifact(
                chunk_output,
                producer_sha256=producer,
                expected_shape=expected_shape,
            ):
                print(
                    f"reused RealBasicVSR chunk {index + 1}/{len(vsr_chunks)}",
                    flush=True,
                )
            else:
                print(f"RealBasicVSR chunk {index + 1}/{len(vsr_chunks)}", flush=True)
                response = vsr_worker.run(request)
                _validate_worker_shape(
                    response,
                    frames=chunk.model_frames,
                    width=spec.width * config.vsr.spatial_scale,
                    height=spec.height * config.vsr.spatial_scale,
                )
                write_npy_receipt(chunk_output, producer_sha256=producer)
                new_vsr_chunks += 1
                if (
                    checkpoint_after_vsr_chunks is not None
                    and new_vsr_chunks >= checkpoint_after_vsr_chunks
                    and index + 1 < len(vsr_chunks)
                ):
                    raise RuntimeError(
                        "requested checkpoint stop after completed RealBasicVSR chunks: "
                        f"new_chunks={new_vsr_chunks}, work_dir={work_dir}"
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
        write_completed_run_manifest(
            output_path.with_name(f"{output_path.stem}.run.json"),
            plan=run_plan,
            output_sha256=sha256_file(result),
            output_size_bytes=result.stat().st_size,
            elapsed_seconds=time.perf_counter() - started_at,
        )
        return result
    finally:
        if not finalized:
            encoder.abort()


def _run_reverse_pipeline(
    repo_root: Path,
    input_path: Path,
    output_path: Path,
    config: PipelineConfig,
    runtime: dict[str, Path],
    work_dir: Path,
    *,
    resume: bool,
    checkpoint_after_vsr_chunks: int | None,
) -> Path:
    """research A/B용 RealBasicVSR → Practical-RIFE 경로를 실행합니다."""

    started_at = time.perf_counter()
    if resume:
        raise ValueError("resume is not yet supported for the research reverse pipeline")
    if checkpoint_after_vsr_chunks is not None:
        raise ValueError("checkpoint stop is only supported for the forward pipeline")
    if config.vfi.backend_id != "practical-rife-v4.25":
        raise ValueError("reverse pipeline requires practical-rife-v4.25")
    if config.vsr.backend_id != "mmagic-realbasicvsr":
        raise ValueError("reverse pipeline requires mmagic-realbasicvsr")
    if config.order.value != "vsr_then_vfi":
        raise ValueError("reverse pipeline requires vsr_then_vfi")
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
    max_vsr_frames = RealBasicVSRResolutionContract.max_model_frames(
        width=spec.width,
        height=spec.height,
    )
    vsr_chunks = plan_realbasicvsr_chunks(
        frame_count=cfr_plan.output_frames,
        cut_after=scene.cut_after,
        max_source_frames=max_vsr_frames,
    )
    rife_chunks = plan_rife_chunks(
        frame_count=cfr_plan.output_frames,
        cut_after=scene.cut_after,
        multiplier=config.vfi.temporal_multiplier,
    )
    run_plan = RunPlanManifest(
        schema_version=2,
        input_path=str(input_path),
        output_path=str(output_path),
        input_sha256=sha256_file(input_path),
        config_sha256=fingerprint_text(config.model_dump_json()),
        order=config.order.value,
        source_width=spec.width,
        source_height=spec.height,
        source_frames=spec.frame_count,
        cfr_frames=cfr_plan.output_frames,
        cfr_dropped_frames=cfr_plan.dropped_frames,
        cfr_duplicated_frames=cfr_plan.duplicated_frames,
        target_fps=_fraction_string(config.cfr.target_fps),
        scene_cut_after=scene.cut_after,
        vfi_backend_id=config.vfi.backend_id,
        vfi_multiplier=config.vfi.temporal_multiplier,
        rife_chunk_count=len(rife_chunks),
        vsr_backend_id=config.vsr.backend_id,
        vsr_scale=config.vsr.spatial_scale,
        vsr_max_model_frames=max_vsr_frames,
        vsr_chunk_count=len(vsr_chunks),
        output_fps=_fraction_string(timeline.output_fps),
        output_frames=timeline.output_frames,
    )
    write_run_plan(work_dir / "run-plan.json", run_plan)
    print(
        f"reverse preflight: source={spec.width}x{spec.height}/"
        f"{cfr_plan.output_frames} CFR frames, cuts={len(scene.cut_after)}, "
        f"VSR chunks={len(vsr_chunks)}, RIFE chunks={len(rife_chunks)}",
        flush=True,
    )

    vsr_input_dir = work_dir / "reverse-vsr-input"
    vsr_assembler = RealBasicVSRInputAssembler(
        vsr_chunks,
        width=spec.width,
        height=spec.height,
        output_dir=vsr_input_dir,
    )
    stream_cfr_rgb24_frames(
        runtime["ffmpeg"],
        input_path,
        width=spec.width,
        height=spec.height,
        plan=cfr_plan,
        color=color,
        consume_frame=vsr_assembler.consume,
    )
    vsr_input_paths = vsr_assembler.finalize()
    worker_environment = {"PYTHONPATH": str(repo_root / "src")}
    vsr_output_dir = work_dir / "reverse-vsr-output"
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
                job_id=f"reverse-realbasicvsr-{index:06d}",
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
            print(f"reverse RealBasicVSR {index + 1}/{len(vsr_chunks)}", flush=True)
            response = vsr_worker.run(request)
            _validate_worker_shape(
                response,
                frames=chunk.model_frames,
                width=spec.width * config.vsr.spatial_scale,
                height=spec.height * config.vsr.spatial_scale,
            )
            vsr_output_paths.append(chunk_output)

    scaled_width = spec.width * config.vsr.spatial_scale
    scaled_height = spec.height * config.vsr.spatial_scale
    rife_input_dir = work_dir / "reverse-rife-input"
    rife_assembler = RifeInputAssembler(
        rife_chunks,
        width=scaled_width,
        height=scaled_height,
        output_dir=rife_input_dir,
    )
    stream_realbasicvsr_output_frames(
        vsr_chunks,
        tuple(vsr_output_paths),
        width=spec.width,
        height=spec.height,
        output_scale=config.vsr.spatial_scale,
        consume_frame=rife_assembler.consume,
    )
    rife_input_paths = rife_assembler.finalize()
    shutil.rmtree(vsr_input_dir)
    shutil.rmtree(vsr_output_dir)

    rife_output_dir = work_dir / "reverse-rife-output"
    rife_output_dir.mkdir(parents=True, exist_ok=False)
    rife_output_paths: list[Path | None] = []
    for index, (chunk, chunk_input) in enumerate(
        zip(rife_chunks, rife_input_paths, strict=True)
    ):
        if not chunk.use_model:
            rife_output_paths.append(None)
            continue
        chunk_output = rife_output_dir / f"rife-output-{index:06d}.npy"
        request = WorkerRequest.create(
            job_id=f"reverse-rife-{index:06d}",
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
        print(f"reverse RIFE {index + 1}/{len(rife_chunks)}", flush=True)
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
            width=scaled_width,
            height=scaled_height,
        )
        rife_output_paths.append(chunk_output)

    encode_contract = EncodeContract.create(
        input_path=input_path,
        output_path=output_path,
        final_output_root=config.runtime.final_output_root,
        width=scaled_width,
        height=scaled_height,
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
        stream_rife_output_frames(
            rife_chunks,
            rife_input_paths,
            tuple(rife_output_paths),
            width=scaled_width,
            height=scaled_height,
            consume_frame=encoder.consume,
        )
        result = encoder.finalize(runtime["ffprobe"])
        finalized = True
        write_provenance_manifest(
            output_path.with_name(f"{output_path.stem}.provenance.json"),
            (
                MODEL_PROVENANCE[config.vsr.backend_id],
                MODEL_PROVENANCE[config.vfi.backend_id],
            ),
        )
        write_completed_run_manifest(
            output_path.with_name(f"{output_path.stem}.run.json"),
            plan=run_plan,
            output_sha256=sha256_file(result),
            output_size_bytes=result.stat().st_size,
            elapsed_seconds=time.perf_counter() - started_at,
        )
        return result
    finally:
        if not finalized:
            encoder.abort()


def _run_pipeline(
    repo_root: Path,
    input_path: Path,
    output_path: Path,
    config: PipelineConfig,
    runtime: dict[str, Path],
    work_dir: Path,
    *,
    resume: bool,
    checkpoint_after_vsr_chunks: int | None,
) -> Path:
    if config.order.value == "vfi_then_vsr":
        return _run_forward_pipeline(
            repo_root,
            input_path,
            output_path,
            config,
            runtime,
            work_dir,
            resume=resume,
            checkpoint_after_vsr_chunks=checkpoint_after_vsr_chunks,
        )
    return _run_reverse_pipeline(
        repo_root,
        input_path,
        output_path,
        config,
        runtime,
        work_dir,
        resume=resume,
        checkpoint_after_vsr_chunks=checkpoint_after_vsr_chunks,
    )


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
    resume = arguments.resume_work is not None
    if arguments.resume_work is None:
        work_dir = Path(tempfile.mkdtemp(prefix="enhance-", dir=jobs_root))
    else:
        if not arguments.resume_work.is_absolute():
            raise ValueError("--resume-work must be an absolute path")
        work_dir = arguments.resume_work.resolve(strict=True)
        if not work_dir.is_dir() or not work_dir.is_relative_to(jobs_root.resolve()):
            raise ValueError(f"--resume-work must be a job directory under {jobs_root}")
    succeeded = False
    try:
        result = _run_pipeline(
            repo_root,
            input_path,
            output_path,
            config,
            runtime,
            work_dir,
            resume=resume,
            checkpoint_after_vsr_chunks=arguments.checkpoint_after_vsr_chunks,
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
