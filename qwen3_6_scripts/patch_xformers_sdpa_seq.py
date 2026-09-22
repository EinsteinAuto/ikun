"""
策略：顺序（per-sequence）fallback — 纯 PyTorch 数学实现
==========================================================
逐条序列用 matmul + softmax 手写 attention，完全绕开所有硬件
flash attention kernel（ixformer / cudnnFlashAttnForward）。

背景：
  Iluvatar cudnnFlashAttnForward 存在两个已知问题：
    1. 不支持 is_causal=True（报错）
    2. 使用 attn_mask 路径时数值结果不正确（静默错误，输出全为"!"）
  与华为昇腾 910B4 上 llama.cpp --flash-attn off 修复同类问题的原理相同。
  纯数学路径（matmul + softmax）在任何 PyTorch 后端上结果都正确。

优点：
  数值正确，不依赖任何硬件特定 attention kernel。
  峰值显存 = max(seq_len)² × H × dtype_size，由 --max-model-len 控制。

缺点：
  并发请求的 prefill attention 串行执行。
  O(L²) 显存（无 flash attention 的 O(L) 优化）。

内存参考（fp16，H_local=6）：
  max-model-len=4096  → 峰值 ~200 MB
  max-model-len=8192  → 峰值 ~800 MB
  max-model-len=16384 → 峰值 ~3.2 GB

额外 patch（arg_utils.py）：
  vllm 0.6.3 在 max_model_len > 32K 时会自动开启 chunked prefill（无命令行
  关闭选项），原意是防止 profiling OOM。但 _run_sdpa_fallback 已通过 Q-tiling
  解决了该问题，chunked prefill 反而会把推理路径从 _run_sdpa_fallback 切换到
  _forward_prefix_pytorch，属于不必要的行为变更，因此一并禁用该自动逻辑。

Deploy:
  python3 modified_scripts/patch_xformers_sdpa_seq.py
"""

from patch_utils import package_root, replace_one_of, replace_once

VLLM_ROOT = package_root("vllm")
XFORMERS_PATH = VLLM_ROOT / "attention" / "backends" / "xformers.py"
ARG_UTILS_PATH = VLLM_ROOT / "engine" / "arg_utils.py"
LOGITS_PROC_PATH = (
    VLLM_ROOT / "model_executor" / "layers" / "logits_processor.py")
OUTLINES_DECODING_PATH = (
    VLLM_ROOT / "model_executor" / "guided_decoding" /
    "outlines_decoding.py")

# _apply_logits_processors crashes when seq_groups is None (intermediate
# chunked-prefill chunks on the driver rank). Add an early-return guard.
_LP_OLD_BLOCK = """\
def _apply_logits_processors(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    found_logits_processors = False\
"""

_LP_NEW_BLOCK = """\
def _apply_logits_processors(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    if sampling_metadata.seq_groups is None:  # intermediate chunked-prefill chunk
        return logits
    found_logits_processors = False\
"""

# Outlines' UNESCAPED_STRING accepts raw JSON control characters, including
# newlines and tabs. The generated text can therefore satisfy the CFG while
# still failing json.loads(). Use the RFC 8259 string character constraints.
_JSON_STRING_OLD_BLOCK = """\
| UNESCAPED_STRING
| SIGNED_NUMBER      -> number
| "true"             -> true
| "false"            -> false
| "null"             -> null

array  : "[" [value ("," value)*] "]"
object : "{" [pair ("," pair)*] "}"
pair   : UNESCAPED_STRING ":" value

%import common.UNESCAPED_STRING
%import common.SIGNED_NUMBER
%import common.WS

%ignore WS\
"""

_JSON_STRING_V1_BLOCK = r'''| JSON_STRING
| SIGNED_NUMBER      -> number
| "true"             -> true
| "false"            -> false
| "null"             -> null

array  : "[" [value ("," value)*] "]"
object : "{" [pair ("," pair)*] "}"
pair   : JSON_STRING ":" value

JSON_STRING: /"(\\["\\\/bfnrt]|\\u[0-9a-fA-F]{4}|[^"\\\x00-\x1f])*"/
%import common.SIGNED_NUMBER
%import common.WS

%ignore WS'''

_JSON_STRING_NEW_BLOCK = r'''| JSON_STRING
| SIGNED_NUMBER      -> number
| "true"             -> true
| "false"            -> false
| "null"             -> null

array  : "[" _ws [value (_ws "," _ws value)*] _ws "]"
object : "{" _ws [pair (_ws "," _ws pair)*] _ws "}"
pair   : JSON_STRING _ws ":" _ws value
_ws    : JSON_WS?

JSON_STRING: /"(\\["\\\/bfnrt]|\\u[0-9a-fA-F]{4}|[^"\\\x00-\x1f])*"/
JSON_WS: /[ \t\r\n]{1,4}/
%import common.SIGNED_NUMBER'''

# vllm 0.6.3 自动开启 chunked prefill 的原始块
_ARG_OLD_BLOCK = """\
                if (is_gpu and not use_sliding_window and not use_spec_decode
                        and not self.enable_lora
                        and not self.enable_prompt_adapter):
                    self.enable_chunked_prefill = True
                    logger.warning(
                        "Chunked prefill is enabled by default for models with "
                        "max_model_len > 32K. Currently, chunked prefill might "
                        "not work with some features or models. If you "
                        "encounter any issues, please disable chunked prefill "
                        "by setting --enable-chunked-prefill=False.")\
"""

_ARG_NEW_BLOCK = """\
                if (is_gpu and not use_sliding_window and not use_spec_decode
                        and not self.enable_lora
                        and not self.enable_prompt_adapter):
                    pass  # skip auto-enable: Q-tiling in _run_sdpa_fallback
                          # handles long-context memory without chunked prefill\
"""

_MM_PREFIX_OLD_BLOCK = """\
        if model_config.is_multimodal_model:
            if self.enable_prefix_caching:
                logger.warning(
                    "--enable-prefix-caching is currently not "
                    "supported for multimodal models and has been disabled.")
            self.enable_prefix_caching = False\
"""

_MM_PREFIX_NEW_BLOCK = """\
        if model_config.is_multimodal_model:
            architectures = getattr(model_config.hf_config,
                                    "architectures", []) or []
            qwen36_native_vision = "Qwen3_5MoeForCausalLM" in architectures
            if self.enable_prefix_caching and qwen36_native_vision:
                logger.info(
                    "Keeping prefix caching enabled for the Qwen3.6 native "
                    "vision path.")
            elif self.enable_prefix_caching:
                logger.warning(
                    "--enable-prefix-caching is currently not "
                    "supported for multimodal models and has been disabled.")
                self.enable_prefix_caching = False\
"""

FALLBACK_METHOD = '''
    def _run_sdpa_fallback(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: "XFormersMetadata",
    ) -> torch.Tensor:
        """纯数学 causal attention fallback，带 Q-tiling 内存优化。

        调用时机：kv_cache.numel()==0（profiling 阶段）。
        此路径无 KV 缓存前缀，KV 长度 == query 长度。

        内存优化（Q-tiling，与 Flash Attention 同思路）：
          将 Q 分成 _Q_CHUNK 大小的子块逐块计算，每块峰值内存
          O(_Q_CHUNK × q_len) 而非 O(q_len²)。
          profiling 阶段序列可能达到 max_model_len（如 20K tokens），
          不加 Q-tiling 会产生 9.6 GB 矩阵直接 OOM。

        softmax 在 float32 下计算以防止 float16 溢出，结果转回原始 dtype。

        Args:
            query : [1, total_query_tokens, num_heads,    head_dim]
            key   : [1, total_query_tokens, num_kv_heads, head_dim]
            value : [1, total_query_tokens, num_kv_heads, head_dim]
        Returns:
            [1, total_query_tokens, num_heads, head_dim]
        """
        _Q_CHUNK = 256  # 与 _forward_prefix_pytorch 的 _ATTN_Q_CHUNK 保持一致

        assert attn_metadata.seq_lens is not None
        orig_dtype = query.dtype
        num_seqs = len(attn_metadata.seq_lens)

        # 推导每条序列的实际 query 长度。
        # 正常 prefill 时 q_len == seq_len；如果将来遇到 chunked 场景，
        # query_start_loc 记录的是真实 query token 数（非全序列长度）。
        if (attn_metadata.query_start_loc is not None
                and len(attn_metadata.query_start_loc) == num_seqs + 1):
            q_lens = [
                int(attn_metadata.query_start_loc[i + 1].item()) -
                int(attn_metadata.query_start_loc[i].item())
                for i in range(num_seqs)
            ]
        else:
            q_lens = list(attn_metadata.seq_lens)

        q_flat = query.squeeze(0)   # [T, H,   D]
        k_flat = key.squeeze(0)     # [T, Hkv, D]
        v_flat = value.squeeze(0)

        output = torch.empty_like(q_flat)
        seq_start = 0
        for q_len in q_lens:
            seq_end = seq_start + q_len

            # 当前序列的完整 K/V（此路径无前缀，KV == Q）
            k_s = k_flat[seq_start:seq_end].permute(1, 0, 2).float()  # [Hkv, q_len, D]
            v_s = v_flat[seq_start:seq_end].permute(1, 0, 2).float()  # [Hkv, q_len, D]

            # GQA：展开 KV heads 至与 query heads 一致
            if k_s.shape[0] != self.num_heads:
                n = self.num_heads // k_s.shape[0]
                k_s = k_s.repeat_interleave(n, dim=0).contiguous()
                v_s = v_s.repeat_interleave(n, dim=0).contiguous()

            # k_pos 用于因果掩码
            k_pos = torch.arange(q_len, device=query.device)

            # Q-tiling：分块处理 query，峰值内存 O(_Q_CHUNK × q_len)
            for qc_start in range(0, q_len, _Q_CHUNK):
                qc_end = min(qc_start + _Q_CHUNK, q_len)

                # [H, qc, D]
                q_c = q_flat[seq_start + qc_start:seq_start + qc_end] \
                      .permute(1, 0, 2).float()

                # [H, qc, q_len]
                attn_w = torch.matmul(q_c, k_s.transpose(-2, -1)) * self.scale

                # 因果掩码：q_c 里位置 j 只能看 k_pos <= j（相对位置）
                qc_q_pos = torch.arange(qc_start, qc_end, device=query.device)
                mask = k_pos.unsqueeze(0) > qc_q_pos.unsqueeze(1)
                attn_w = attn_w.masked_fill(mask.unsqueeze(0), float("-inf"))

                attn_w = torch.softmax(attn_w, dim=-1)
                out_c = torch.matmul(attn_w, v_s).to(orig_dtype)  # [H, qc, D]

                output[seq_start + qc_start:seq_start + qc_end] = (
                    out_c.permute(1, 0, 2))

            seq_start = seq_end

        return output.unsqueeze(0)  # [1, T, H, D]

'''

OLD_XFORMER_BLOCK = """\
        self.attn_op = xops.fmha.flash.FwOp()
        if self.alibi_slopes is None:
            # Add the batch dimension.
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            out = xops.memory_efficient_attention_forward(
                query,
                key,
                value,
                attn_bias=attn_bias[0],
                p=0.0,
                scale=self.scale,
                op = self.attn_op
                )
            return out.view_as(original_query)\
"""

NEW_XFORMER_BLOCK = """\
        self.attn_op = xops.fmha.flash.FwOp()
        if self.alibi_slopes is None:
            # Add the batch dimension.
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            if self.head_size > 128:
                out = self._run_sdpa_fallback(query, key, value, attn_metadata)
            else:
                out = xops.memory_efficient_attention_forward(
                    query,
                    key,
                    value,
                    attn_bias=attn_bias[0],
                    p=0.0,
                    scale=self.scale,
                    op=self.attn_op,
                )
            return out.view_as(original_query)\
"""

INJECT_ANCHOR = "    def _run_memory_efficient_xformers_forward("

_PREFIX_CALL_OLD_BLOCK = """\
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
                )\
"""

_PREFIX_CALL_NEW_BLOCK = """\
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
                )\
"""


def patch_file(path):
    replace_once(
        path,
        INJECT_ANCHOR,
        FALLBACK_METHOD + INJECT_ANCHOR,
        required=True,
        already_contains="def _run_sdpa_fallback(")
    replace_once(
        path,
        OLD_XFORMER_BLOCK,
        NEW_XFORMER_BLOCK,
        required=True,
        already_contains="out = self._run_sdpa_fallback(query, key, value, attn_metadata)")
    replace_once(
        path,
        _PREFIX_CALL_OLD_BLOCK,
        _PREFIX_CALL_NEW_BLOCK,
        required=True,
        already_contains=(
            "is_causal_decoder=(attn_type == AttentionType.DECODER)"))


def patch_arg_utils(path):
    replace_once(
        path,
        _ARG_OLD_BLOCK,
        _ARG_NEW_BLOCK,
        required=True,
        already_contains="skip auto-enable: Q-tiling")
    replace_once(
        path,
        _MM_PREFIX_OLD_BLOCK,
        _MM_PREFIX_NEW_BLOCK,
        required=True,
        already_contains="Keeping prefix caching enabled for the Qwen3.6")


def patch_logits_processor(path):
    replace_once(
        path,
        _LP_OLD_BLOCK,
        _LP_NEW_BLOCK,
        required=True,
        already_contains="intermediate chunked-prefill chunk")


def patch_outlines_json_grammar(path):
    replace_one_of(
        path,
        [
            (_JSON_STRING_V1_BLOCK, _JSON_STRING_NEW_BLOCK),
            (_JSON_STRING_OLD_BLOCK, _JSON_STRING_NEW_BLOCK),
        ],
        required=True,
        already_contains="JSON_WS:")


def main():
    print("=== patch_xformers_sdpa_seq (sequential, pure-math) ===")
    print(f"Target: {XFORMERS_PATH}")
    patch_file(XFORMERS_PATH)

    print("\n=== patch_arg_utils (disable chunked-prefill auto-enable) ===")
    print(f"Target: {ARG_UTILS_PATH}")
    patch_arg_utils(ARG_UTILS_PATH)

    print("\n=== patch_logits_processor (seq_groups=None guard for chunked prefill) ===")
    print(f"Target: {LOGITS_PROC_PATH}")
    patch_logits_processor(LOGITS_PROC_PATH)

    print("\n=== patch_outlines_json_grammar (reject raw control chars) ===")
    print(f"Target: {OUTLINES_DECODING_PATH}")
    patch_outlines_json_grammar(OUTLINES_DECODING_PATH)

    print("\nDone.")


if __name__ == "__main__":
    main()
