from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from rvfi_sr.config_loader import load_hydra_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


class HydraConfigLoaderTest(unittest.TestCase):
    def test_default_and_deterministic_presets_compose_and_validate(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            root = Path(temporary_directory)
            input_directory = root / "input"
            input_directory.mkdir()
            overrides = (
                f"input_dir={input_directory}",
                f"output_dir={root / 'output'}",
                "runtime.final_output_root=null",
            )
            default_config = load_hydra_config(CONFIG_DIR, "default", overrides)
            deterministic_config = load_hydra_config(
                CONFIG_DIR,
                "deterministic",
                overrides,
            )
            reverse_config = load_hydra_config(
                CONFIG_DIR,
                "reverse_deterministic",
                overrides,
            )
            self.assertEqual(default_config.vsr.backend_id, "flashvsr-v1.1")
            self.assertEqual(
                deterministic_config.vsr.backend_id,
                "mmagic-realbasicvsr",
            )
            self.assertEqual(reverse_config.order.value, "vsr_then_vfi")
            self.assertEqual(reverse_config.vsr.backend_id, "mmagic-realbasicvsr")

    def test_research_preset_is_inert_until_license_opt_in(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_directory:
            root = Path(temporary_directory)
            input_directory = root / "input"
            input_directory.mkdir()
            overrides = (
                f"input_dir={input_directory}",
                f"output_dir={root / 'output'}",
                "runtime.final_output_root=null",
            )
            with self.assertRaisesRegex(ValidationError, "restricted license"):
                load_hydra_config(CONFIG_DIR, "research_quality", overrides)
            config = load_hydra_config(
                CONFIG_DIR,
                "research_quality",
                (*overrides, "runtime.allow_restricted_license=true"),
            )
            self.assertEqual(config.vfi.backend_id, "bim-vfi")


if __name__ == "__main__":
    unittest.main()
