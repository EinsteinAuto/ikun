#!/usr/bin/env python3
"""
patch_chat_template.py — Fix non-thinking mode '!!!!!' output

Root cause: Qwen3.5/3.6-MoE's chat_template adds '<think>\n\n</think>\n\n'
when enable_thinking=false. This empty think block causes the model to
degenerate into outputting nothing but '!'.

Fix: Remove the empty think block so the model generates directly.

Verified on real machine 2026-08-20:
  - Model: /root/public-storage/models/Qwen/Qwen3.6-35B-A3B
  - target repr: "{{- '<think>\\n\\n</think>\\n\\n' }}"
  - match: True → patch applied successfully
"""
import json
import sys
import os


def patch_tokenizer_config(model_path: str) -> bool:
    config_path = os.path.join(model_path, "tokenizer_config.json")
    if not os.path.isfile(config_path):
        print(f"[patch_chat_template] {config_path} not found")
        return False

    with open(config_path, "r") as f:
        config = json.load(f)

    template = config.get("chat_template", "")
    if not template:
        print("[patch_chat_template] No chat_template found")
        return False

    # The Jinja template contains literal backslash-n sequences: \n
    # After json.load these remain as two-char sequences (backslash + n),
    # NOT real newlines. Use raw string so Python doesn't interpret them.
    target = r"{{- '<think>\n\n</think>\n\n' }}"
    replacement = "{{- '' }}"

    if target in template:
        template = template.replace(target, replacement, 1)
        config["chat_template"] = template
        with open(config_path, "w") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print("[patch_chat_template] ✓ Removed empty <think></think> block for non-thinking mode")
        return True

    # Fallback: check if already patched
    if "enable_thinking is false" in template and target not in template:
        print("[patch_chat_template] Already patched or different format")
        return True

    print(f"[patch_chat_template] WARNING: Could not find target pattern")
    # Debug: show what's actually there
    idx = template.find("enable_thinking is false")
    if idx >= 0:
        print(f"[patch_chat_template] Context: {repr(template[idx:idx+150])}")
    return False


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_PATH", "/model")
    success = patch_tokenizer_config(model_path)
    sys.exit(0 if success else 1)