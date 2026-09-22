"""
Patches the vLLM model registry and deploys the Qwen3_5 model file.

Deploy steps on the remote machine:
  1. patch_ops.sh locates vLLM with importlib.util.find_spec.
  2. cp modified_scripts/qwen3_5.py into the detected vllm model directory.
  2. python3 modified_scripts/patch_vllm_qwen3_5.py

The registry patch installs Qwen3.6 aliases so /model/config.json does not
need to be edited by hand.
"""

import ast

from patch_utils import package_root, replace_once

VLLM_ROOT = package_root("vllm")
REGISTRY = VLLM_ROOT / "model_executor" / "models" / "registry.py"
MODEL = VLLM_ROOT / "model_executor" / "models" / "qwen3_5.py"

EXPECTED_REGISTRY_ENTRIES = (
    '"Qwen3ForCausalLM": ("qwen3_5", "Qwen3_5ForCausalLM")',
    '"Qwen3MoeForCausalLM": ("qwen3_5", "Qwen3_5MoeForCausalLM")',
    '"Qwen3_5ForCausalLM": ("qwen3_5", "Qwen3_5ForCausalLM")',
    '"Qwen3_5MoeForCausalLM": ("qwen3_5", "Qwen3_5MoeForCausalLM")',
    '"Qwen3_6ForCausalLM": ("qwen3_5", "Qwen3_5ForCausalLM")',
    '"Qwen3_6MoeForCausalLM": ("qwen3_5", "Qwen3_5MoeForCausalLM")',
)


def main():
    print(f"=== Patching {REGISTRY} ===")
    replace_once(
        REGISTRY,
        '    "Qwen3ForCausalLM": ("qwen3", "Qwen3ForCausalLM"),\n'
        '    "Qwen3MoeForCausalLM": ("qwen3_moe", "Qwen3MoeForCausalLM"),',
        '    "Qwen3ForCausalLM": ("qwen3_5", "Qwen3_5ForCausalLM"),\n'
        '    "Qwen3MoeForCausalLM": ("qwen3_5", "Qwen3_5MoeForCausalLM"),\n'
        '    "Qwen3_5ForCausalLM": ("qwen3_5", "Qwen3_5ForCausalLM"),\n'
        '    "Qwen3_5MoeForCausalLM": ("qwen3_5", "Qwen3_5MoeForCausalLM"),\n'
        '    "Qwen3_6ForCausalLM": ("qwen3_5", "Qwen3_5ForCausalLM"),\n'
        '    "Qwen3_6MoeForCausalLM": ("qwen3_5", "Qwen3_5MoeForCausalLM"),',
        required=True,
        already_contains='"Qwen3_6MoeForCausalLM"')

    print("\n=== Static verification ===")
    model_source = MODEL.read_text(encoding="utf-8")
    tree = ast.parse(model_source, filename=str(MODEL))
    class_names = {
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    }
    required_classes = {"Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM"}
    missing_classes = required_classes - class_names
    if missing_classes:
        raise RuntimeError(
            f"Qwen3.5 model classes missing: {sorted(missing_classes)}")

    registry_source = REGISTRY.read_text(encoding="utf-8")
    missing_entries = [
        entry for entry in EXPECTED_REGISTRY_ENTRIES
        if entry not in registry_source
    ]
    if missing_entries:
        raise RuntimeError(
            f"Qwen3.5 registry entries missing: {missing_entries}")
    print("  model syntax and class declarations verified without import")
    print(f"  registry aliases verified: {len(EXPECTED_REGISTRY_ENTRIES)}")

    print("\nDone. Registry aliases installed; do not edit /model/config.json.")


if __name__ == "__main__":
    main()
