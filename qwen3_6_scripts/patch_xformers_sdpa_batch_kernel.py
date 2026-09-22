"""
策略：批量（block-diagonal）— F.scaled_dot_product_attention，可走硬件 kernel
=============================================================================
构建块对角 causal mask，对整批序列一次 F.scaled_dot_product_attention。
与 patch_xformers_sdpa_batch.py（纯 matmul）的区别：
  SDPA 会根据 PyTorch/驱动能力分发到最优 kernel（Flash Attention /
  mem-efficient attention / math fallback），而不是固定走 cublas matmul。

历史说明：
  该方案最早因输出全"!"而被弃用，后续排查确认"!"由 mamba_cache.py bug
  引起，与 attention 实现无关。当前恢复此方案用于性能对比测试。

已知硬件限制（BI-V100）：
  cudnnFlashAttnForward 不支持 is_causal=True（报错）。
  本实现使用 is_causal=False + 显式块对角 additive mask 规避此限制。
  若 SDPA 仍分发到有问题的 kernel，回退到 patch_xformers_sdpa_batch.py。

优点（vs 纯 matmul）：
  SDPA 可分发到 Flash Attention kernel → O(L) 显存、更快的 CUDA kernel。

缺点：
  依赖硬件 kernel 行为，若 kernel 有 bug 则数值错误（需与 matmul 版对比验证）。

Deploy:
  python3 modified_scripts/patch_xformers_sdpa_batch_kernel.py
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
        """批量 F.scaled_dot_product_attention fallback（可走硬件 kernel）。

        构建块对角 causal mask，对整批序列一次 SDPA 调用。
        SDPA 可分发到 Flash Attention / mem-efficient attention kernel。
        is_causal=False + 显式 additive mask，规避 cudnnFlashAttnForward
        不支持 is_causal=True 的限制。

        块对角 mask（seq1 len=3，seq2 len=2）：
                 s1,0  s1,1  s1,2  s2,0  s2,1
          s1,0 [  0   -inf  -inf  -inf  -inf ]
          s1,1 [  0    0   -inf  -inf  -inf ]
          s1,2 [  0    0    0   -inf  -inf ]
          s2,0 [-inf  -inf  -inf   0   -inf ]
          s2,1 [-inf  -inf  -inf   0    0  ]

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
        total_tokens = query.shape[1]

        # ── 块对角 causal mask [T, T] ─────────────────────────────────────
        mask = torch.full(
            (total_tokens, total_tokens),
            float("-inf"),
            dtype=orig_dtype,
            device=query.device,
        )
        start = 0
        for seq_len in attn_metadata.seq_lens:
            end = start + seq_len
            mask[start:end, start:end] = torch.tril(
                torch.zeros(seq_len, seq_len, dtype=orig_dtype, device=query.device)
            )
            start = end

        # ── [1, H, T, D] ──────────────────────────────────────────────────
        q_all = query.squeeze(0).permute(1, 0, 2).contiguous().unsqueeze(0)
        k_all = key.squeeze(0).permute(1, 0, 2).contiguous().unsqueeze(0)
        v_all = value.squeeze(0).permute(1, 0, 2).contiguous().unsqueeze(0)

        # ── GQA：展开 KV heads ────────────────────────────────────────────
        if k_all.shape[1] != q_all.shape[1]:
            n = q_all.shape[1] // k_all.shape[1]
            k_all = k_all.repeat_interleave(n, dim=1).contiguous()
            v_all = v_all.repeat_interleave(n, dim=1).contiguous()

        # ── F.scaled_dot_product_attention（可走硬件 kernel）─────────────
        # is_causal=False：避免 cudnnFlashAttnForward "not support causal mode"
        # attn_mask 传 additive float mask（非 bool），SDPA 选择 math/kernel 路径
        out = F.scaled_dot_product_attention(
            q_all, k_all, v_all,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
        )
        # [1, H, T, D] → [1, T, H, D]
        return out.squeeze(0).permute(1, 0, 2).contiguous().unsqueeze(0)

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
    print("=== patch_xformers_sdpa_batch_kernel (batch, F.sdpa + kernel dispatch) ===")
    print(f"Target: {XFORMERS_PATH}")
    patch_file(XFORMERS_PATH)
    print("\nDone.")


if __name__ == "__main__":
    main()
