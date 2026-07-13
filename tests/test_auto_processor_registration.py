"""Regression tests for custom AutoProcessor registration."""

import ast
from pathlib import Path


def _models_dir():
    return Path(__file__).resolve().parents[1] / "mlx_vlm" / "models"


def _literal_model_types(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.List):
        values = set()
        for item in node.elts:
            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                values.add(item.value)
        return values
    return set()


def test_models_do_not_register_auto_processors_with_model_type_strings():
    repo_root = Path(__file__).resolve().parents[1]
    models_dir = _models_dir()
    offenders = []

    for path in models_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "register":
                continue
            if not isinstance(node.func.value, ast.Name):
                continue
            if node.func.value.id != "AutoProcessor":
                continue
            if node.args and isinstance(node.args[0], ast.Constant):
                if isinstance(node.args[0].value, str):
                    offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}")

    assert offenders == []


def test_issue_1564_processors_use_model_type_patch():
    expected_model_types = {
        "deepseek_vl_v2",
        "deepseekocr",
        "deepseekocr_2",
        "glm4v",
        "glm4v_moe",
        "jina_vlm",
        "unlimited-ocr",
    }
    patched_model_types = set()

    for path in _models_dir().rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                is_patch_call = node.func.id == "install_auto_processor_patch"
            elif isinstance(node.func, ast.Attribute):
                is_patch_call = node.func.attr == "install_auto_processor_patch"
            else:
                is_patch_call = False
            if is_patch_call and node.args:
                patched_model_types.update(_literal_model_types(node.args[0]))

    assert expected_model_types <= patched_model_types
