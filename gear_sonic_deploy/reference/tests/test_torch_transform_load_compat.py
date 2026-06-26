import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TORCH_TRANSFORM_PATH = REPO_ROOT / "gear_sonic" / "trl" / "utils" / "torch_transform.py"


class TorchTransformLoadCompatTest(unittest.TestCase):
    def test_compute_human_joints_loads_local_pickle_with_weights_only_false(self):
        tree = ast.parse(TORCH_TRANSFORM_PATH.read_text())
        compute_func = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "compute_human_joints"
        )
        torch_load_calls = [
            node
            for node in ast.walk(compute_func)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
        ]

        self.assertEqual(len(torch_load_calls), 1)
        weights_only_keywords = [
            keyword for keyword in torch_load_calls[0].keywords if keyword.arg == "weights_only"
        ]
        self.assertEqual(len(weights_only_keywords), 1)
        self.assertIs(weights_only_keywords[0].value.value, False)


if __name__ == "__main__":
    unittest.main()
