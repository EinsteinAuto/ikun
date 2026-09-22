"""
策略：批量（block-diagonal）fallback — 纯 PyTorch 数学实现
=============================================================
构建块对角 causal mask，对整批序列一次 matmul + softmax，
完全绕开所有硬件 flash attention kernel。

背景：
  ixformer flshattF:         head_dim > 128 报错拒绝
  cudnnFlashAttnForward:     接受 head_dim=256，但数值结果错误（输出全"!"）
  两者大概率是同一硬件单元，ixformer 提前拦截了硬件不支持的配置。
  纯 matmul 路径完全绕开硬件 flash attention，数值正确。

优点：
  数值正确。
  并发请求 prefill attention 在 GPU 上真正并行（一次大 matmul）。

缺点：
  峰值显存 = total_tokens² × H × dtype_size
  total_tokens 受 --max-num-batched-tokens 控制，max-model-len 控制不住。

内存参考（fp16，H_local=6，--max-num-batched-tokens=T）：
  T=2048  → 峰值 ~50  MB
  T=4096  → 峰值 ~200 MB
  T=8192  → 峰值 ~800 MB
  T=16384 → 峰值 ~3.2 GB

Deploy:
  python3 modified_scripts/patch_xformers_sdpa_batch.py
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
        """批量纯数学 attention fallback。

        构建块对角 causal mask（等价于 ixformer BlockDiagonalCausalMask），
        对整批序列一次 matmul + softmax，GPU 并行处理所有序列。

        块对角 mask 结构（seq1 len=3，seq2 len=2）：
                 s1,0  s1,1  s1,2  s2,0  s2,1
          s1,0 [  0   -inf  -inf  -inf  -inf ]
          s1,1 [  0    0   -inf  -inf  -inf ]
          s1,2 [  0    0    0   -inf  -inf ]
          s2,0 [-inf  -inf  -inf   0   -inf ]
          s2,1 [-inf  -inf  -inf   0    0  ]

        softmax 在 float32 下计算防止 float16 溢出，结果转回原始 dtype。

        Args:
            query : [1, total_prefill_tokens, num_heads,    head_dim]
            key   : [1, total_prefill_tokens, num_kv_heads, head_dim]
            value : [1, total_prefill_tokens, num_kv_heads, head_dim]
        Returns:
            [1, total_prefill_tokens, num_heads, head_dim]
        """
        assert attn_metadata.seq_lens is not None
        orig_dtype = query.dtype
        total_tokens = query.shape[1]

        # ── 构建块对角 causal mask [T, T] ────────────────────────────────
        # 全部初始化为 -inf，再对每条序列的对角块填入下三角 0
        mask = torch.full(
            (total_tokens, total_tokens),
            float("-inf"),
            dtype=torch.float32,
            device=query.device,
        )
        start = 0
        for seq_len in attn_metadata.seq_lens:
            end = start + seq_len
            mask[start:end, start:end] = torch.tril(
                torch.zeros(seq_len, seq_len,
                            dtype=torch.float32, device=query.device)
            )
            start = end

        # ── [1, H, T, D]，.contiguous() ──────────────────────────────────
        q_all = query.squeeze(0).permute(1, 0, 2).contiguous().unsqueeze(0)
        k_all = key.squeeze(0).permute(1, 0, 2).contiguous().unsqueeze(0)
        v_all = value.squeeze(0).permute(1, 0, 2).contiguous().unsqueeze(0)

        # ── GQA：展开 KV heads ────────────────────────────────────────────
        if k_all.shape[1] != q_all.shape[1]:
            n = q_all.shape[1] // k_all.shape[1]
            k_all = k_all.repeat_interleave(n, dim=1).contiguous()
            v_all = v_all.repeat_interleave(n, dim=1).contiguous()

        # ── 纯数学 attention（float32 防溢出）────────────────────────────
        # [1, H, T, T]
        attn_w = torch.matmul(q_all.float(), k_all.float().transpose(-2, -1))
        attn_w = attn_w * self.scale
        attn_w = attn_w + mask  # 加法广播：mask [T,T] → [1, H, T, T]
        attn_w = torch.softmax(attn_w, dim=-1)

        out = torch.matmul(attn_w, v_all.float()).to(orig_dtype)
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
    print("=== patch_xformers_sdpa_batch (batch, pure-math) ===")
    print(f"Target: {XFORMERS_PATH}")
    patch_file(XFORMERS_PATH)
    print("\nDone.")


if __name__ == "__main__":
    main()
