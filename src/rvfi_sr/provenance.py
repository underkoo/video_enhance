"""모델 코드와 체크포인트의 재현 가능한 출처 계약을 정의합니다."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


def _validate_https_url(name: str, value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"{name} must be an absolute HTTPS URL")


@dataclass(frozen=True, slots=True)
class CheckpointArtifact:
    """다운로드 전에 크기와 digest를 확정한 checkpoint 파일입니다."""

    filename: str
    url: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if (
            not self.filename.strip()
            or Path(self.filename).name != self.filename
            or self.filename in {".", ".."}
        ):
            raise ValueError("filename must be a safe basename")
        _validate_https_url("url", self.url)
        if not _SHA256_PATTERN.fullmatch(self.sha256):
            raise ValueError("sha256 must be a lowercase 64-character digest")
        if isinstance(self.size_bytes, bool) or self.size_bytes < 1:
            raise ValueError("size_bytes must be a positive integer")


@dataclass(frozen=True, slots=True)
class ModelProvenance:
    """실행에 사용한 upstream 코드와 checkpoint의 불변 식별자입니다."""

    backend_id: str
    upstream_url: str
    upstream_commit: str
    license_name: str
    checkpoints: tuple[CheckpointArtifact, ...]

    def __post_init__(self) -> None:
        if not self.backend_id.strip():
            raise ValueError("backend_id must not be empty")
        _validate_https_url("upstream_url", self.upstream_url)
        if not _COMMIT_PATTERN.fullmatch(self.upstream_commit):
            raise ValueError("upstream_commit must be a lowercase 40-character Git SHA")
        if not self.license_name.strip():
            raise ValueError("license_name must not be empty")
        if not self.checkpoints:
            raise ValueError("checkpoints must not be empty")
        if any(not isinstance(item, CheckpointArtifact) for item in self.checkpoints):
            raise TypeError("checkpoints must contain only CheckpointArtifact")
        filenames = [item.filename for item in self.checkpoints]
        if len(filenames) != len(set(filenames)):
            raise ValueError("checkpoint filenames must be unique")


def verify_checkpoint(path: Path, spec: CheckpointArtifact) -> None:
    """checkpoint 파일명, 크기, SHA-256을 순서대로 검증합니다."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if not isinstance(spec, CheckpointArtifact):
        raise TypeError("spec must be CheckpointArtifact")
    resolved_path = path.resolve(strict=True)
    if not resolved_path.is_file():
        raise FileNotFoundError(resolved_path)
    if resolved_path.name != spec.filename:
        raise ValueError(
            f"checkpoint filename mismatch: expected={spec.filename!r}, "
            f"actual={resolved_path.name!r}"
        )
    actual_size = resolved_path.stat().st_size
    if actual_size != spec.size_bytes:
        raise ValueError(
            f"checkpoint size mismatch: expected={spec.size_bytes}, actual={actual_size}"
        )

    digest = hashlib.sha256()
    with resolved_path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != spec.sha256:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected={spec.sha256}, actual={actual_sha256}"
        )

def write_provenance_manifest(
    output_path: Path,
    models: tuple[ModelProvenance, ...],
) -> None:
    """출처 manifest를 partial 파일에 기록한 뒤 원자적으로 확정합니다."""

    if not isinstance(output_path, Path):
        raise TypeError("output_path must be pathlib.Path")
    if output_path.suffix.casefold() != ".json":
        raise ValueError("output_path must use the .json suffix")
    if not models:
        raise ValueError("models must not be empty")
    if any(not isinstance(model, ModelProvenance) for model in models):
        raise TypeError("models must contain only ModelProvenance")

    resolved_output = output_path.resolve(strict=False)
    if resolved_output.exists():
        raise FileExistsError(resolved_output)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    partial_path = resolved_output.with_name(
        f"{resolved_output.stem}.partial{resolved_output.suffix}"
    )
    if partial_path.exists():
        raise FileExistsError(partial_path)

    payload = {
        "schema_version": 1,
        "models": [asdict(model) for model in models],
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    try:
        with partial_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial_path, resolved_output)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise
