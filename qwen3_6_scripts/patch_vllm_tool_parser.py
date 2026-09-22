"""
Patches vLLM 0.6.3 to register Qwen3CoderToolParser under the name "qwen3_coder".

Deploy steps on the remote machine (already called by patch_ops.sh):
  1. patch_ops.sh locates vLLM with importlib.util.find_spec.
  2. cp qwen3coder_tool_parser.py into the detected vllm tool_parsers.
  2. python3 patch_vllm_tool_parser.py

Usage after patching:
  --tool-call-parser qwen3_coder --enable-auto-tool-choice
"""

from patch_utils import ensure_dir, package_root, replace_once

VLLM_ROOT = package_root("vllm")
TOOL_PARSERS_DIR = VLLM_ROOT / "entrypoints" / "openai" / "tool_parsers"
INIT_FILE = TOOL_PARSERS_DIR / "__init__.py"


def main():
    ensure_dir(TOOL_PARSERS_DIR)

    print(f"=== Patching {INIT_FILE} ===")
    replace_once(
        INIT_FILE,
        "from .mistral_tool_parser import MistralToolParser",
        "from .mistral_tool_parser import MistralToolParser\n"
        "from .qwen3coder_tool_parser import Qwen3CoderToolParser",
        required=True,
        already_contains="from .qwen3coder_tool_parser import Qwen3CoderToolParser")
    replace_once(
        INIT_FILE,
        '"MistralToolParser", "Internlm2ToolParser", "Llama3JsonToolParser"\n]',
        '"MistralToolParser", "Internlm2ToolParser", "Llama3JsonToolParser",\n'
        '    "Qwen3CoderToolParser"\n]',
        required=True,
        already_contains='"Qwen3CoderToolParser"')

    print("\n=== Verification ===")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "qwen3coder_tool_parser",
            str(TOOL_PARSERS_DIR / "qwen3coder_tool_parser.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        print(f"  Module spec loaded: {spec.name}")
        print("  (full import requires torch/vllm runtime — skipping exec)")
    except Exception as e:
        print(f"  [optional] spec check failed: {e}")

    print("\nDone. Start vLLM server with:")
    print("  --tool-call-parser qwen3_coder --enable-auto-tool-choice")


if __name__ == "__main__":
    main()
