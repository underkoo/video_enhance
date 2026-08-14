"""ffprobe JSON을 엄격한 미디어 명세로 변환합니다."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any


@dataclass(frozen=True, slots=True)
class ColorMetadata:
    """디코딩과 재인코딩 사이에 보존할 색상 메타데이터입니다."""

    pixel_format: str
    range: str | None
    space: str | None
    transfer: str | None
    primaries: str | None

    @property
    def is_complete(self) -> bool:
        """재인코딩에 필요한 네 color tag가 모두 기록됐는지 반환합니다."""

        return all(
            value is not None
            for value in (self.range, self.space, self.transfer, self.primaries)
        )


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    """원본 오디오 stream 메타데이터입니다."""

    stream_index: int
    codec: str
    sample_rate: int
    channels: int
    duration: Fraction


@dataclass(frozen=True, slots=True)
class MediaSpec:
    """한 입력 파일의 추론 및 검증용 미디어 명세입니다."""

    format_name: str
    video_stream_index: int
    video_codec: str
    width: int
    height: int
    average_fps: Fraction
    nominal_fps: Fraction
    time_base: Fraction
    frame_count: int
    video_duration: Fraction
    duration: Fraction
    color: ColorMetadata
    audio: AudioMetadata | None

    @property
    def requires_cfr_normalization(self) -> bool:
        """컨테이너 rate 표기가 불일치하면 CFR 정규화를 요구합니다."""

        return self.average_fps != self.nominal_fps


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence")
    return value


def _string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_string(mapping: Mapping[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string when present")
    return value.strip()


def _integer(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool):
        raise TypeError(f"{key} must be an integer")
    if isinstance(value, int):
        return value
    if value is None:
        raise ValueError(f"{key} must be present")
    if not isinstance(value, str):
        raise TypeError(f"{key} must be an integer or integer string")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer") from exc


def _positive_fraction(mapping: Mapping[str, Any], key: str) -> Fraction:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a rational string")
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{key} must be a valid rational string") from exc
    if parsed <= 0:
        raise ValueError(f"{key} must be positive")
    return parsed


def parse_ffprobe_payload(payload: Mapping[str, object]) -> MediaSpec:
    """`-count_frames -show_streams -show_format` JSON을 검증합니다."""

    root = _mapping(payload, "payload")
    format_payload = _mapping(root.get("format"), "format")
    stream_payloads = _sequence(root.get("streams"), "streams")
    streams = [_mapping(stream, "stream") for stream in stream_payloads]
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(videos) != 1:
        raise ValueError(f"expected exactly one video stream, found {len(videos)}")
    if len(audios) > 1:
        raise ValueError(f"expected at most one audio stream, found {len(audios)}")

    video = videos[0]
    width = _integer(video, "width")
    height = _integer(video, "height")
    frame_count = _integer(video, "nb_read_frames")
    if width <= 0:
        raise ValueError("width must be positive")
    if height <= 0:
        raise ValueError("height must be positive")
    if frame_count <= 0:
        raise ValueError("nb_read_frames must be positive")

    audio: AudioMetadata | None = None
    if audios:
        audio_payload = audios[0]
        sample_rate = _integer(audio_payload, "sample_rate")
        channels = _integer(audio_payload, "channels")
        if sample_rate <= 0 or channels <= 0:
            raise ValueError("audio sample_rate and channels must be positive")
        audio = AudioMetadata(
            stream_index=_integer(audio_payload, "index"),
            codec=_string(audio_payload, "codec_name"),
            sample_rate=sample_rate,
            channels=channels,
            duration=_positive_fraction(audio_payload, "duration"),
        )

    return MediaSpec(
        format_name=_string(format_payload, "format_name"),
        video_stream_index=_integer(video, "index"),
        video_codec=_string(video, "codec_name"),
        width=width,
        height=height,
        average_fps=_positive_fraction(video, "avg_frame_rate"),
        nominal_fps=_positive_fraction(video, "r_frame_rate"),
        time_base=_positive_fraction(video, "time_base"),
        frame_count=frame_count,
        video_duration=_positive_fraction(video, "duration"),
        duration=_positive_fraction(format_payload, "duration"),
        color=ColorMetadata(
            pixel_format=_string(video, "pix_fmt"),
            range=_optional_string(video, "color_range"),
            space=_optional_string(video, "color_space"),
            transfer=_optional_string(video, "color_transfer"),
            primaries=_optional_string(video, "color_primaries"),
        ),
        audio=audio,
    )
