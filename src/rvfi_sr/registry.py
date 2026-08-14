"""공식 소스를 검토한 모델 백엔드의 정적 기능 registry입니다."""

from __future__ import annotations

from rvfi_sr.backends import BackendCapabilities, BackendKind, LicenseUse

BACKEND_REGISTRY: dict[str, BackendCapabilities] = {
    "practical-rife-v4.25": BackendCapabilities(
        backend_id="practical-rife-v4.25",
        kind=BackendKind.VFI,
        license_use=LicenseUse.PERMISSIVE,
        python_spec=">=3.10,<3.12",
        alignment=128,
        temporal_multipliers=(2, 4, 8),
    ),
    "gimm-vfi-r-p": BackendCapabilities(
        backend_id="gimm-vfi-r-p",
        kind=BackendKind.VFI,
        license_use=LicenseUse.NON_COMMERCIAL,
        python_spec=">=3.7,<3.8",
        alignment=32,
        temporal_multipliers=(2, 4, 8),
    ),
    "bim-vfi": BackendCapabilities(
        backend_id="bim-vfi",
        kind=BackendKind.VFI,
        license_use=LicenseUse.RESEARCH_ONLY,
        python_spec=">=3.11,<3.12",
        alignment=32,
        temporal_multipliers=(2, 4, 8),
    ),
    "flashvsr-v1.1": BackendCapabilities(
        backend_id="flashvsr-v1.1",
        kind=BackendKind.VSR,
        license_use=LicenseUse.PERMISSIVE,
        python_spec=">=3.11,<3.12",
        alignment=16,
        spatial_scales=(4,),
    ),
    "mmagic-realbasicvsr": BackendCapabilities(
        backend_id="mmagic-realbasicvsr",
        kind=BackendKind.VSR,
        license_use=LicenseUse.PERMISSIVE,
        python_spec=">=3.9,<3.12",
        alignment=1,
        spatial_scales=(4,),
    ),
}


def get_backend_capabilities(backend_id: str) -> BackendCapabilities:
    """등록되지 않은 문자열 fallback 없이 정확한 backend를 반환합니다."""

    try:
        return BACKEND_REGISTRY[backend_id]
    except KeyError as error:
        raise ValueError(f"unknown backend_id: {backend_id!r}") from error
