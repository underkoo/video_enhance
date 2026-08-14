from __future__ import annotations

import unittest

from rvfi_sr.preflight import GpuSpec, validate_gpu


class GpuPreflightTest(unittest.TestCase):
    def test_parses_nvidia_smi_csv_without_locale_dependent_units(self) -> None:
        gpu = GpuSpec.from_nvidia_smi_csv("0, NVIDIA GeForce RTX 3090, 24576, 8.6\n")
        self.assertEqual(gpu.index, 0)
        self.assertEqual(gpu.name, "NVIDIA GeForce RTX 3090")
        self.assertEqual(gpu.memory_total_mib, 24_576)
        self.assertEqual(gpu.compute_capability, (8, 6))

    def test_rejects_malformed_or_multiple_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            GpuSpec.from_nvidia_smi_csv("")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            GpuSpec.from_nvidia_smi_csv(
                "0, NVIDIA GeForce RTX 3090, 24576, 8.6\n"
                "1, NVIDIA GeForce RTX 3090, 24576, 8.6\n"
            )
        with self.assertRaisesRegex(ValueError, "compute capability"):
            GpuSpec.from_nvidia_smi_csv("0, GPU, 24576, unknown\n")

    def test_enforces_backend_vram_and_compute_requirements(self) -> None:
        gpu = GpuSpec(
            index=0,
            name="NVIDIA GeForce RTX 3090",
            memory_total_mib=24_576,
            compute_capability=(8, 6),
        )
        validate_gpu(gpu, expected_index=0, minimum_memory_mib=23_000, minimum_compute=(8, 6))
        with self.assertRaisesRegex(RuntimeError, "VRAM"):
            validate_gpu(
                gpu,
                expected_index=0,
                minimum_memory_mib=30_000,
                minimum_compute=(8, 6),
            )
        with self.assertRaisesRegex(RuntimeError, "compute capability"):
            validate_gpu(
                gpu,
                expected_index=0,
                minimum_memory_mib=23_000,
                minimum_compute=(9, 0),
            )


if __name__ == "__main__":
    unittest.main()
