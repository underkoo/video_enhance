#!/usr/bin/env python3
"""동일 input의 VFI→VSR / VSR→VFI 결과를 GT 없는 proxy로 비교합니다."""

from __future__ import annotations

import argparse
import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import numpy as np

from rvfi_sr.cfr import probe_cfr_plan, stream_cfr_rgb24_frames
from rvfi_sr.color import RgbDecodeContract, VideoColorRange, VideoColorSpace
from rvfi_sr.media_tools import run_ffprobe
from rvfi_sr.quality_metrics import OrderQualityAccumulator, VariantQualityMetrics
from rvfi_sr.receipts import sha256_file


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--forward", required=True, type=Path)
    parser.add_argument("--reverse", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-stride", type=int, default=2)
    return parser.parse_args()


def _absolute_mp4(path: Path, name: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.suffix.casefold() != ".mp4":
        raise ValueError(f"{name} must be an existing MP4")
    return resolved


def _load_plan(video_path: Path) -> dict[str, Any]:
    manifest_path = video_path.with_name(f"{video_path.stem}.run.json")
    payload = json.loads(manifest_path.resolve(strict=True).read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 2
        or not isinstance(payload.get("plan"), dict)
        or not isinstance(payload.get("result"), dict)
    ):
        raise TypeError(f"invalid run manifest: {manifest_path}")
    plan = cast(dict[str, Any], payload["plan"])
    result = cast(dict[str, Any], payload["result"])
    if plan.get("schema_version") != 2:
        raise ValueError(f"unsupported run plan schema: {manifest_path}")
    actual_sha256 = sha256_file(video_path)
    if result.get("output_sha256") != actual_sha256:
        raise RuntimeError(f"run manifest output SHA-256 mismatch: {video_path}")
    if result.get("output_size_bytes") != video_path.stat().st_size:
        raise RuntimeError(f"run manifest output size mismatch: {video_path}")
    return plan


def _validate_plans(
    input_path: Path,
    forward_path: Path,
    reverse_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    forward = _load_plan(forward_path)
    reverse = _load_plan(reverse_path)
    if forward.get("order") != "vfi_then_vsr":
        raise ValueError("forward manifest order must be vfi_then_vsr")
    if reverse.get("order") != "vsr_then_vfi":
        raise ValueError("reverse manifest order must be vsr_then_vfi")
    actual_input_sha256 = sha256_file(input_path)
    if forward.get("input_sha256") != actual_input_sha256:
        raise RuntimeError("forward manifest input SHA-256 mismatch")
    if reverse.get("input_sha256") != actual_input_sha256:
        raise RuntimeError("reverse manifest input SHA-256 mismatch")
    comparable_fields = (
        "input_path",
        "source_width",
        "source_height",
        "source_frames",
        "cfr_frames",
        "cfr_dropped_frames",
        "cfr_duplicated_frames",
        "target_fps",
        "scene_cut_after",
        "vfi_backend_id",
        "vfi_multiplier",
        "vsr_backend_id",
        "vsr_scale",
        "output_fps",
        "output_frames",
    )
    differences = tuple(
        field for field in comparable_fields if forward.get(field) != reverse.get(field)
    )
    if differences:
        raise RuntimeError(f"A/B run plans are not comparable: fields={differences}")
    return forward, reverse


def _decode_source(
    ffmpeg_path: Path,
    ffprobe_path: Path,
    input_path: Path,
    plan: dict[str, Any],
) -> np.ndarray[Any, np.dtype[np.uint8]]:
    spec = run_ffprobe(ffprobe_path, input_path)
    target_fps = Fraction(str(plan["target_fps"]))
    cfr = probe_cfr_plan(
        ffmpeg_path,
        input_path,
        expected_source_frames=spec.frame_count,
        target_fps=target_fps,
    )
    if cfr.output_frames != plan["cfr_frames"]:
        raise RuntimeError("source CFR frame count differs from benchmark run plan")
    color = RgbDecodeContract.create(
        spec.color,
        untagged_range=VideoColorRange.TV,
        untagged_space=VideoColorSpace.BT709,
    )
    frames = np.empty((cfr.output_frames, spec.height, spec.width, 3), dtype=np.uint8)

    def consume(frame_index: int, frame: bytes) -> None:
        frames[frame_index] = np.frombuffer(frame, dtype=np.uint8).reshape(
            spec.height,
            spec.width,
            3,
        )

    stream_cfr_rgb24_frames(
        ffmpeg_path,
        input_path,
        width=spec.width,
        height=spec.height,
        plan=cfr,
        color=color,
        consume_frame=consume,
    )
    return frames


def _measure_variant(
    ffmpeg_path: Path,
    ffprobe_path: Path,
    video_path: Path,
    source_frames: np.ndarray[Any, np.dtype[np.uint8]],
    plan: dict[str, Any],
    *,
    sample_stride: int,
) -> tuple[VariantQualityMetrics, dict[str, int | str]]:
    spec = run_ffprobe(ffprobe_path, video_path)
    expected_geometry = (source_frames.shape[2] * 2, source_frames.shape[1] * 2)
    if (spec.width, spec.height) != expected_geometry:
        raise RuntimeError(
            f"variant geometry mismatch: expected={expected_geometry}, "
            f"actual={(spec.width, spec.height)}"
        )
    output_fps = Fraction(str(plan["output_fps"]))
    cfr = probe_cfr_plan(
        ffmpeg_path,
        video_path,
        expected_source_frames=spec.frame_count,
        target_fps=output_fps,
    )
    if (
        cfr.output_frames != plan["output_frames"]
        or cfr.dropped_frames != 0
        or cfr.duplicated_frames != 0
    ):
        raise RuntimeError("enhanced variant is not an exact CFR benchmark timeline")
    color = RgbDecodeContract.create(
        spec.color,
        untagged_range=VideoColorRange.TV,
        untagged_space=VideoColorSpace.BT709,
    )
    cut_after = tuple(int(value) for value in plan["scene_cut_after"])
    accumulator = OrderQualityAccumulator(
        source_frames,
        output_width=spec.width,
        output_height=spec.height,
        cut_after=cut_after,
        sample_stride=sample_stride,
    )
    stream_cfr_rgb24_frames(
        ffmpeg_path,
        video_path,
        width=spec.width,
        height=spec.height,
        plan=cfr,
        color=color,
        consume_frame=accumulator.consume,
    )
    return accumulator.finalize(), {
        "sha256": sha256_file(video_path),
        "size_bytes": video_path.stat().st_size,
        "width": spec.width,
        "height": spec.height,
        "frame_count": spec.frame_count,
        "fps": f"{spec.average_fps.numerator}/{spec.average_fps.denominator}",
    }


def _write_atomic_json(output_path: Path, payload: dict[str, object]) -> Path:
    if not output_path.is_absolute() or output_path.suffix.casefold() != ".json":
        raise ValueError("--output must be an absolute JSON path")
    if not output_path.is_relative_to(Path("/mnt/d")):
        raise ValueError("benchmark output must stay under /mnt/d")
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")
    if partial.exists():
        raise FileExistsError(partial)
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, output_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return output_path


def main() -> int:
    arguments = _arguments()
    input_path = _absolute_mp4(arguments.input, "--input")
    forward_path = _absolute_mp4(arguments.forward, "--forward")
    reverse_path = _absolute_mp4(arguments.reverse, "--reverse")
    if isinstance(arguments.sample_stride, bool) or arguments.sample_stride < 1:
        raise ValueError("--sample-stride must be positive")
    forward_plan, reverse_plan = _validate_plans(
        input_path,
        forward_path,
        reverse_path,
    )
    repo_root = Path(__file__).resolve().parents[1]
    ffmpeg_path = (
        repo_root / ".runtime/tools/ffmpeg-n8.1.2-34-g9b6c8969e0/bin/ffmpeg"
    )
    ffprobe_path = (
        repo_root / ".runtime/tools/ffmpeg-n8.1.2-34-g9b6c8969e0/bin/ffprobe"
    )
    source_frames = _decode_source(
        ffmpeg_path,
        ffprobe_path,
        input_path,
        forward_plan,
    )
    forward_metrics, forward_artifact = _measure_variant(
        ffmpeg_path,
        ffprobe_path,
        forward_path,
        source_frames,
        forward_plan,
        sample_stride=arguments.sample_stride,
    )
    reverse_metrics, reverse_artifact = _measure_variant(
        ffmpeg_path,
        ffprobe_path,
        reverse_path,
        source_frames,
        reverse_plan,
        sample_stride=arguments.sample_stride,
    )
    forward_values = forward_metrics.to_dict()
    reverse_values = reverse_metrics.to_dict()
    deltas = {
        key: reverse_values[key] - forward_values[key]
        for key in forward_values
        if isinstance(forward_values[key], float)
        and isinstance(reverse_values[key], float)
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "input": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
        },
        "sample_stride": arguments.sample_stride,
        "metric_scope": (
            "GT-free proxies; lower is smoother for temporal/curvature/overshoot, "
            "higher is sharper for Laplacian and more source-faithful for PSNR"
        ),
        "forward_vfi_then_vsr": {
            "artifact": forward_artifact,
            "metrics": forward_values,
        },
        "reverse_vsr_then_vfi": {
            "artifact": reverse_artifact,
            "metrics": reverse_values,
        },
        "delta_reverse_minus_forward": deltas,
        "automatic_winner": None,
    }
    result = _write_atomic_json(arguments.output.resolve(strict=False), payload)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
