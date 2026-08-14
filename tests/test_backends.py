from __future__ import annotations

import unittest

from rvfi_sr.backends import (
    BackendCapabilities,
    BackendKind,
    BackendRequest,
    LicenseUse,
)


def make_vfi_capabilities() -> BackendCapabilities:
    return BackendCapabilities(
        backend_id="practical-rife-v4.25",
        kind=BackendKind.VFI,
        license_use=LicenseUse.PERMISSIVE,
        python_spec=">=3.10,<3.12",
        alignment=32,
        temporal_multipliers=(2, 4, 8),
    )


class BackendCapabilitiesTest(unittest.TestCase):
    def test_vfi_accepts_only_declared_temporal_multiplier(self) -> None:
        capabilities = make_vfi_capabilities()
        capabilities.validate_request(BackendRequest(temporal_multiplier=4))
        with self.assertRaisesRegex(ValueError, "temporal multiplier"):
            capabilities.validate_request(BackendRequest(temporal_multiplier=3))

    def test_vfi_rejects_spatial_request(self) -> None:
        with self.assertRaisesRegex(ValueError, "spatial_scale"):
            make_vfi_capabilities().validate_request(BackendRequest(spatial_scale=2))

    def test_vsr_native_scale_is_accepted(self) -> None:
        capabilities = BackendCapabilities(
            backend_id="flashvsr-v1.1",
            kind=BackendKind.VSR,
            license_use=LicenseUse.PERMISSIVE,
            python_spec=">=3.11,<3.12",
            alignment=16,
            spatial_scales=(4,),
        )
        capabilities.validate_request(BackendRequest(spatial_scale=4))

    def test_vsr_post_downsample_requires_explicit_opt_in_and_integer_ratio(self) -> None:
        capabilities = BackendCapabilities(
            backend_id="flashvsr-v1.1",
            kind=BackendKind.VSR,
            license_use=LicenseUse.PERMISSIVE,
            python_spec=">=3.11,<3.12",
            alignment=16,
            spatial_scales=(4,),
        )
        with self.assertRaisesRegex(ValueError, "post_downsample"):
            capabilities.validate_request(BackendRequest(spatial_scale=2))
        capabilities.validate_request(
            BackendRequest(spatial_scale=2, post_downsample=True)
        )
        with self.assertRaisesRegex(ValueError, "integer ratio"):
            capabilities.validate_request(
                BackendRequest(spatial_scale=3, post_downsample=True)
            )

    def test_restricted_license_requires_explicit_opt_in(self) -> None:
        capabilities = BackendCapabilities(
            backend_id="gimm-vfi-r-p",
            kind=BackendKind.VFI,
            license_use=LicenseUse.NON_COMMERCIAL,
            python_spec=">=3.7,<3.8",
            alignment=32,
            temporal_multipliers=(2, 4, 8),
        )
        with self.assertRaisesRegex(PermissionError, "restricted license"):
            capabilities.validate_request(BackendRequest(temporal_multiplier=2))
        capabilities.validate_request(
            BackendRequest(
                temporal_multiplier=2,
                allow_restricted_license=True,
            )
        )

    def test_capability_shape_is_kind_specific(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not declare spatial"):
            BackendCapabilities(
                backend_id="invalid-vfi",
                kind=BackendKind.VFI,
                license_use=LicenseUse.PERMISSIVE,
                python_spec=">=3.10,<3.12",
                alignment=32,
                temporal_multipliers=(2,),
                spatial_scales=(4,),
            )


if __name__ == "__main__":
    unittest.main()
