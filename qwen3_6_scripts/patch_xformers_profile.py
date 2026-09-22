"""Install disabled-by-default M1-48 XFormers timing boundaries."""

from __future__ import annotations

from pathlib import Path

try:
    from patch_utils import package_root, replace_once
except ModuleNotFoundError:
    from .patch_utils import package_root, replace_once


IMPORT_OLD = "from vllm.logger import init_logger"
IMPORT_NEW = """\
from vllm.bi100_profile import bi100_timer
from vllm.logger import init_logger"""

KV_WRITE_OLD = """\
                PagedAttention.write_to_paged_cache(key, value, key_cache,
                                                    value_cache,
                                                    updated_slot_mapping,
                                                    self.kv_cache_dtype,
                                                    k_scale, v_scale)"""
KV_WRITE_NEW = """\
                with bi100_timer("xformers.kv_write"):
                    PagedAttention.write_to_paged_cache(
                        key, value, key_cache, value_cache,
                        updated_slot_mapping, self.kv_cache_dtype,
                        k_scale, v_scale)"""

DENSE_OLD = """\
                out = self._run_memory_efficient_xformers_forward(
                    query, key, value, prefill_meta, attn_type=attn_type)"""
DENSE_NEW = """\
                with bi100_timer("xformers.dense_prefill"):
                    out = self._run_memory_efficient_xformers_forward(
                        query, key, value, prefill_meta, attn_type=attn_type)"""

PAGED_OLD = """\
                out = PagedAttention.forward_prefix(
                    query,
                    key,
                    value,
                    self.kv_cache_dtype,
                    key_cache,
                    value_cache,
                    prefill_meta.block_tables,
                    prefill_meta.query_start_loc,
                    prefill_meta.seq_lens_tensor,
                    prefill_meta.context_lens_tensor,
                    prefill_meta.max_query_len,
                    self.alibi_slopes,
                    self.sliding_window,
                    k_scale,
                    v_scale,
                    is_causal_decoder=(attn_type == AttentionType.DECODER),
                )"""
PAGED_NEW = """\
                with bi100_timer("xformers.paged_prefill"):
                    out = PagedAttention.forward_prefix(
                        query,
                        key,
                        value,
                        self.kv_cache_dtype,
                        key_cache,
                        value_cache,
                        prefill_meta.block_tables,
                        prefill_meta.query_start_loc,
                        prefill_meta.seq_lens_tensor,
                        prefill_meta.context_lens_tensor,
                        prefill_meta.max_query_len,
                        self.alibi_slopes,
                        self.sliding_window,
                        k_scale,
                        v_scale,
                        is_causal_decoder=(attn_type == AttentionType.DECODER),
                    )"""


def patch_file(path: Path) -> None:
    replace_once(
        path,
        IMPORT_OLD,
        IMPORT_NEW,
        already_contains="from vllm.bi100_profile import bi100_timer",
    )
    replace_once(
        path,
        KV_WRITE_OLD,
        KV_WRITE_NEW,
        already_contains='bi100_timer("xformers.kv_write")',
    )
    replace_once(
        path,
        DENSE_OLD,
        DENSE_NEW,
        already_contains='bi100_timer("xformers.dense_prefill")',
    )
    replace_once(
        path,
        PAGED_OLD,
        PAGED_NEW,
        already_contains='bi100_timer("xformers.paged_prefill")',
    )
    text = path.read_text(encoding="utf-8")
    canonical = "\n".join(line.rstrip(" \t") for line in text.split("\n"))
    if not canonical.endswith("\n"):
        canonical += "\n"
    if canonical != text:
        path.write_text(canonical, encoding="utf-8")


def main() -> None:
    path = package_root("vllm") / "attention" / "backends" / "xformers.py"
    print("=== patch_xformers_profile (M1-48 diagnostic timers) ===")
    print(f"Target: {path}")
    patch_file(path)


if __name__ == "__main__":
    main()
