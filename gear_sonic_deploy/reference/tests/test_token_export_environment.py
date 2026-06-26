import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALL_SCRIPT = REPO_ROOT / "install_scripts" / "install_token_export.sh"
SETUP_SCRIPT = REPO_ROOT / "gear_sonic_deploy" / "scripts" / "setup_token_export_env.sh"
DOC_PATH = REPO_ROOT / "gear_sonic_deploy" / "reference" / "NYMERIA_RGB_SMPL_TOKEN_EXPORT.md"


class TokenExportEnvironmentTest(unittest.TestCase):
    def test_install_script_creates_dedicated_token_export_venv(self):
        self.assertTrue(INSTALL_SCRIPT.is_file(), f"missing {INSTALL_SCRIPT}")

        text = INSTALL_SCRIPT.read_text()

        self.assertIn(".venv_token_export", text)
        self.assertIn("gear_sonic_token_export", text)
        self.assertIn("https://download.pytorch.org/whl/cpu", text)
        self.assertIn("TOKEN_EXPORT_TORCH_WHEEL", text)
        self.assertIn("TOKEN_EXPORT_TORCH_INDEX_URL", text)
        self.assertIn("TOKEN_EXPORT_TORCH_FALLBACK_INDEX_URL", text)
        self.assertIn("TOKEN_EXPORT_TORCH_SPEC", text)
        self.assertIn("mirrors.aliyun.com/pytorch-wheels/cpu", text)
        self.assertIn('"$TOKEN_EXPORT_TORCH_WHEEL"', text)
        self.assertIn('"$TOKEN_EXPORT_TORCH_SPEC" --index-url "$TOKEN_EXPORT_TORCH_INDEX_URL"', text)
        self.assertIn('"$TOKEN_EXPORT_TORCH_SPEC" --index-url "$TOKEN_EXPORT_TORCH_FALLBACK_INDEX_URL"', text)
        self.assertIn("uv pip install --no-cache \\", text)
        self.assertIn('uv pip install --no-cache -e "gear_sonic" --no-deps', text)
        for name in (
            "PIP_INDEX_URL",
            "PIP_EXTRA_INDEX_URL",
            "UV_INDEX",
            "UV_INDEX_URL",
            "UV_EXTRA_INDEX_URL",
            "UV_DEFAULT_INDEX",
        ):
            self.assertIn(f"unset {name}", text)
        self.assertIn("UV_NO_CONFIG=1", text)
        self.assertIn("onnxruntime", text)
        self.assertNotIn("setup_env.sh", text)
        self.assertNotIn("g1_deploy", text)

    def test_setup_script_activates_dedicated_venv_without_deploy_runtime(self):
        self.assertTrue(SETUP_SCRIPT.is_file(), f"missing {SETUP_SCRIPT}")

        text = SETUP_SCRIPT.read_text()

        self.assertIn(".venv_token_export/bin/activate", text)
        self.assertIn("TOKEN_EXPORT_REPO_ROOT", text)
        self.assertNotIn("setup_env.sh", text)
        self.assertNotIn("FASTRTPS", text)
        self.assertNotIn("TensorRT", text)
        self.assertNotIn("ROS2", text)

    def test_documentation_uses_token_export_environment(self):
        text = DOC_PATH.read_text()

        self.assertIn("bash install_scripts/install_token_export.sh", text)
        self.assertIn("source gear_sonic_deploy/scripts/setup_token_export_env.sh", text)
        self.assertIn("CPU-only PyTorch", text)
        self.assertIn("no-cache", text)
        self.assertIn("TOKEN_EXPORT_TORCH_WHEEL", text)
        self.assertIn("TOKEN_EXPORT_TORCH_INDEX_URL", text)
        self.assertIn("Aliyun", text)
        self.assertNotIn("source scripts/setup_env.sh", text)


if __name__ == "__main__":
    unittest.main()
