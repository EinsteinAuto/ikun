"""
Patches transformers 4.55.3 to register qwen3_5 and qwen3_5_moe model types.

Deploy steps on the remote machine:
  1. patch_ops.sh locates transformers with importlib.util.find_spec.
  2. cp -r modified_scripts/qwen3_5* into the detected transformers/models.
  3. python3 modified_scripts/patch_transformers_qwen3_5.py
"""

import sys

from patch_utils import package_root, replace_once, replace_one_of

TRANSFORMERS_ROOT = package_root("transformers")
AUTO_CONFIG = TRANSFORMERS_ROOT / "models" / "auto" / "configuration_auto.py"
MODELS_INIT = TRANSFORMERS_ROOT / "models" / "__init__.py"


def main():
    print(f"=== Patching {AUTO_CONFIG} ===")
    replace_one_of(AUTO_CONFIG, [
        # CONFIG_MAPPING_NAMES: insert qwen3_5 + qwen3_5_moe right after qwen3
        (
            '("qwen3", "Qwen3Config"),',
            '("qwen3", "Qwen3Config"),\n        ("qwen3_5", "Qwen3_5Config"),\n        ("qwen3_5_moe", "Qwen3_5MoeConfig"),',
        ),
        (
            '("qwen3", "Qwen3Config")\n',
            '("qwen3", "Qwen3Config"),\n        ("qwen3_5", "Qwen3_5Config"),\n        ("qwen3_5_moe", "Qwen3_5MoeConfig"),\n',
        ),
    ], required=True, already_contains='("qwen3_5_moe", "Qwen3_5MoeConfig")')
    replace_one_of(AUTO_CONFIG, [
        # MODEL_NAMES_MAPPING (model_type -> human readable name)
        (
            '("qwen3", "Qwen3"),',
            '("qwen3", "Qwen3"),\n        ("qwen3_5", "Qwen3_5"),\n        ("qwen3_5_moe", "Qwen3_5_MoE"),',
        ),
        (
            '("qwen3", "Qwen3")\n',
            '("qwen3", "Qwen3"),\n        ("qwen3_5", "Qwen3_5"),\n        ("qwen3_5_moe", "Qwen3_5_MoE"),\n',
        ),
    ], required=True, already_contains='("qwen3_5_moe", "Qwen3_5_MoE")')

    print(f"\n=== Patching {MODELS_INIT} ===")
    replace_once(
        MODELS_INIT,
        "from .qwen3 import *\n",
        "from .qwen3 import *\n    from .qwen3_5 import *\n    from .qwen3_5_moe import *\n",
        required=True,
        already_contains="from .qwen3_5_moe import *")

    # Verification
    print("\n=== Verification ===")
    try:
        import importlib.util, types

        def _load_config_mod(module_name, file_path):
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = ".".join(module_name.split(".")[:-1])
            pkg = sys.modules.setdefault("transformers", types.ModuleType("transformers"))
            pkg.__path__ = [str(TRANSFORMERS_ROOT)]
            cu = sys.modules.setdefault(
                "transformers.configuration_utils", types.ModuleType("transformers.configuration_utils"))
            class _PC:
                def __init__(self, **kwargs):
                    return None
            cu.PretrainedConfig = _PC
            for sub in ("transformers.models", f"transformers.models.{module_name.split('.')[-2]}"):
                m = sys.modules.setdefault(sub, types.ModuleType(sub))
                m.__path__ = [str(TRANSFORMERS_ROOT)]
            spec.loader.exec_module(mod)
            return mod

        mod27 = _load_config_mod(
            "transformers.models.qwen3_5.configuration_qwen3_5",
            str(TRANSFORMERS_ROOT / "models" / "qwen3_5" /
                "configuration_qwen3_5.py"),
        )
        cfg = mod27.Qwen3_5Config()
        print(f"  Qwen3_5Config() smoke-test OK     (model_type={cfg.model_type})")

        mod35 = _load_config_mod(
            "transformers.models.qwen3_5_moe.configuration_qwen3_5_moe",
            str(TRANSFORMERS_ROOT / "models" / "qwen3_5_moe" /
                "configuration_qwen3_5_moe.py"),
        )
        moe_cfg = mod35.Qwen3_5MoeConfig()
        print(f"  Qwen3_5MoeConfig() smoke-test OK  (model_type={moe_cfg.model_type})")
        t = moe_cfg.text_config
        print(f"    num_experts={t.num_experts}, top_k={t.num_experts_per_tok}, "
              f"shared={t.shared_expert_intermediate_size}, layers={t.num_hidden_layers}")
    except Exception as e:
        print(f"  [optional] smoke-test failed (may be fine at runtime): {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
