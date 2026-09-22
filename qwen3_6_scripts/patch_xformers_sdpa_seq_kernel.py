"""
策略：顺序（per-sequence）— F.scaled_dot_product_attention，可走硬件 kernel
=============================================================================
逐条序列调用 F.scaled_dot_product_attention，is_causal=False + 显式因果 mask。
与 patch_xformers_sdpa_seq.py（纯 matmul）的区别：
  SDPA 可分发到 Flash Attention / mem-efficient attention kernel，
  而纯 matmul 固定走 cublas。

硬件限制（BI-V100）：
  cudnnFlashAttnForward 不支持 is_causal=True（直接报错）。
  必须使用 is_causal=False + 显式 additive causal mask。
  每条序列单独构造上三角 -inf mask，peak 显存 = max(seq_len)² × dtype，
  比 batch 版的 total_tokens² 小得多。

与 batch_kernel 的对比：
  seq_kernel:   显存小，peak = max_single_seq²；并发 prefill 串行排队
  batch_kernel: 显存大，peak = total_tokens²；并发 prefill 一次并行处理，
                通过 --max-num-batched-tokens 控制 total_tokens 上限

Deploy:
  python3 modified_scripts/patch_xformers_sdpa_seq_kernel.py
"""

from patch_utils import package_root, replace_once

XFORMERS_PATH = package_root("vllm") / "attention" / "backends" / "xformers.py"

FALLBACK_METHOD = '''
    def _run_sdpa_fallback(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: "XFormersMetadata",
    ) -> torch.Tensor:
        """顺序 F.scaled_dot_product_attention fallback（可走硬件 kernel）。

        逐条序列调用 SDPA，is_causal=False + 显式上三角 additive mask。
        cudnnFlashAttnForward 不支持 is_causal=True，必须用显式 mask。
        逐序列构造 mask，peak 显存 = max(seq_len)² × dtype（远小于 batch 版）。

        Args:
            query : [1, total_prefill_tokens, num_heads,    head_dim]
            key   : [1, total_prefill_tokens, num_kv_heads, head_dim]
            value : [1, total_prefill_tokens, num_kv_heads, head_dim]
        Returns:
            [1, total_prefill_tokens, num_heads, head_dim]
        """
        import torch.nn.functional as F

        assert attn_metadata.seq_lens is not None
        orig_dtype = query.dtype

        q_flat = query.squeeze(0)   # [T, H,   D]
        k_flat = key.squeeze(0)     # [T, Hkv, D]
        v_flat = value.squeeze(0)

        output = torch.empty_like(q_flat)
        start = 0
        for seq_len in attn_metadata.seq_lens:
            end = start + seq_len
            # [1, H, L, D]
            q_s = q_flat[start:end].permute(1, 0, 2).contiguous().unsqueeze(0)
            k_s = k_flat[start:end].permute(1, 0, 2).contiguous().unsqueeze(0)
            v_s = v_flat[start:end].permute(1, 0, 2).contiguous().unsqueeze(0)

            # GQA：展开 KV heads
            if k_s.shape[1] != q_s.shape[1]:
                n = q_s.shape[1] // k_s.shape[1]
                k_s = k_s.repeat_interleave(n, dim=1).contiguous()
                v_s = v_s.repeat_interleave(n, dim=1).contiguous()

            # 逐序列因果 mask [L, L]，上三角 -inf
            causal_mask = torch.tril(
                torch.zeros(seq_len, seq_len, dtype=orig_dtype, device=q_s.device)
            )
            causal_mask = causal_mask.masked_fill(
                torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool,
                                     device=q_s.device), diagonal=1),
                float("-inf"),
            )

            # is_causal=False + 显式 mask，规避 cudnnFlashAttnForward 不支持 is_causal=True
            out_s = F.scaled_dot_product_attention(
                q_s, k_s, v_s,
                attn_mask=causal_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scale,
            )
            # [1, H, L, D] → [L, H, D]
            output[start:end] = out_s.squeeze(0).permute(1, 0, 2).to(orig_dtype)
            start = end

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


def main():
    print("=== patch_xformers_sdpa_seq_kernel (seq, F.sdpa + kernel dispatch) ===")
    print(f"Target: {XFORMERS_PATH}")
    patch_file(XFORMERS_PATH)
    print("\nDone.")


if __name__ == "__main__":
    main()
