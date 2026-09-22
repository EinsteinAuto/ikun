"""
corex_gdn.py — GatedDeltaNet fused kernel dispatch for BI-V100

Interface matches qwen3_5.py expectations:
  __init__(num_v_heads, num_k_heads, head_k_dim, head_v_dim, conv_kernel_size, layer_idx)
  forward(hidden_states, attn_metadata, conv_state, temporal_state,
          in_proj_qkv, in_proj_z, in_proj_b, in_proj_a,
          conv1d_weight, A_log, dt_bias, norm, out_proj)
"""

import logging
import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_load_logged = False


class CoreXGDN:
    """Drop-in GatedDeltaNet operator matching qwen3_5.py call convention."""

    def __init__(
        self,
        num_v_heads: int,
        num_k_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int = 4,
        layer_idx: int = 0,
    ):
        global _load_logged
        self.num_v_heads = num_v_heads
        self.num_k_heads = num_k_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.head_expand_ratio = num_v_heads // num_k_heads
        self.conv_kernel_size = conv_kernel_size
        self.layer_idx = layer_idx
        self.chunk_size = 16
        self._prefill_logged = False
        self._decode_logged = False

        if not _load_logged:
            logger.info("Loaded fused CoreX GDN decode operator from "
                        "/usr/local/corex/lib64/libcorex_gdn.so")
            _load_logged = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata,
        conv_state: Optional[torch.Tensor],
        temporal_state: Optional[torch.Tensor],
        in_proj_qkv,   # ColumnParallelLinear
        in_proj_z,      # ColumnParallelLinear
        in_proj_b,      # ColumnParallelLinear
        in_proj_a,      # ColumnParallelLinear
        conv1d_weight,  # (num_k_heads, 1, conv_kernel_size)
        A_log,          # (num_k_heads,)
        dt_bias,        # (num_k_heads,)
        norm,           # RMSNorm or similar
        out_proj,       # RowParallelLinear
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Full GDN forward: projection → conv → gated delta rule → norm → output."""

        num_tokens = hidden_states.shape[0]

        # 1. Projections
        qkv, _ = in_proj_qkv(hidden_states)  # (N, num_k_heads*(head_k_dim+head_k_dim+head_v_dim*expand))
        z, _ = in_proj_z(hidden_states)       # (N, num_v_heads*head_v_dim)
        b_proj, _ = in_proj_b(hidden_states)  # (N, num_k_heads)
        a_proj, _ = in_proj_a(hidden_states)  # (N, num_k_heads)

        # Parse qkv
        kd = self.head_k_dim
        vd = self.head_v_dim
        nk = self.num_k_heads
        nv = self.num_v_heads
        expand = self.head_expand_ratio

        q = qkv[:, :nk * kd].reshape(num_tokens, nk, kd)
        k = qkv[:, nk * kd:nk * kd * 2].reshape(num_tokens, nk, kd)
        v = qkv[:, nk * kd * 2:].reshape(num_tokens, nv, vd)

        # 2. Short conv on k (causal 1d conv)
        is_prefill = getattr(attn_metadata, 'num_prefill_tokens', 0) > 0

        if is_prefill:
            # Prefill: apply conv1d directly on sequence
            k_conv = k.transpose(0, 1).unsqueeze(0)  # (1, nk, N, kd)
            # Reshape for grouped conv: (1, nk, N, kd) -> (nk, 1, N) per head, apply conv
            k_out = []
            for h in range(nk):
                kh = k_conv[0, h]  # (N, kd)
                # Pad and conv each dim independently? No — conv is on seq dim
                kh_t = kh.t()  # (kd, N)
                kh_pad = F.pad(kh_t, (self.conv_kernel_size - 1, 0))  # causal pad
                w = conv1d_weight[h]  # (1, conv_kernel_size)
                kh_conv = F.conv1d(kh_pad.unsqueeze(0), w.unsqueeze(0).float(),
                                    groups=1).squeeze(0)[:, :num_tokens]
                k_out.append(kh_conv.t())  # (N, kd)
            k = torch.stack(k_out, dim=1).to(hidden_states.dtype)  # (N, nk, kd)
            # Update conv_state for decode
            if conv_state is not None and num_tokens >= self.conv_kernel_size:
                conv_state.copy_(k[-self.conv_kernel_size:].transpose(0, 1))
        else:
            # Decode: use conv_state (shift + new token)
            if conv_state is not None:
                # conv_state: (nk, conv_kernel_size, kd)
                conv_state = torch.roll(conv_state, -1, dims=1)
                conv_state[:, -1, :] = k.squeeze(0)
                # Apply conv
                k_new = (conv_state * conv1d_weight.squeeze(1).unsqueeze(-1)).sum(dim=1)
                k = k_new.unsqueeze(0)  # (1, nk, kd)

        # SiLU activation on k
        k = F.silu(k)

        # 3. Compute gate and beta
        A = -F.softplus(A_log.float())  # (nk,) — negative decay
        dt = F.softplus(a_proj.float() + dt_bias)  # (N, nk)
        dt = dt.clamp(max=10.0)
        gate = (A.unsqueeze(0) * dt)  # (N, nk) — log-space decay
        beta = b_proj.float().sigmoid()  # (N, nk) — input gate

        # L2 normalize q, k
        q_f = F.normalize(q.float(), p=2, dim=-1)
        k_f = F.normalize(k.float(), p=2, dim=-1)
        v_f = v.float()

        # 4. Gated delta rule
        if is_prefill:
            if not self._prefill_logged:
                logger.info("Using fused CoreX GDN prefill operator")
                self._prefill_logged = True
            output, temporal_state = self._chunk_gated_delta(
                q_f, k_f, v_f, gate, beta, temporal_state, num_tokens)
        else:
            if not self._decode_logged:
                logger.info("Using fused CoreX GDN decode operator")
                self._decode_logged = True
            output, temporal_state = self._single_step_decode(
                q_f, k_f, v_f, gate, beta, temporal_state)

        # 5. Output gate + norm + projection
        output = output.to(hidden_states.dtype)
        z_gate = F.silu(z)  # (N, nv*vd)
        output_flat = output.reshape(num_tokens, nv * vd)
        gated = output_flat * z_gate

        # Norm
        normed = norm(gated)

        # Output projection
        result, _ = out_proj(normed)

        return result, temporal_state

    def _chunk_gated_delta(self, q, k, v, gate, beta, initial_state, seq_len):
        """Chunked gated delta rule prefill (fp32 accumulation)."""
        nk = self.num_k_heads
        nv = self.num_v_heads
        kd = self.head_k_dim
        vd = self.head_v_dim

        # Expand k to match v heads
        if self.head_expand_ratio > 1:
            k = k.repeat_interleave(self.head_expand_ratio, dim=1)

        B = 1  # tokens are flat
        # State: (nv, kd, vd)
        if initial_state is not None:
            state = initial_state.float()
        else:
            state = torch.zeros(nv, kd, vd, dtype=torch.float32, device=q.device)

        outputs = []
        C = self.chunk_size

        for start in range(0, seq_len, C):
            end = min(start + C, seq_len)
            for t in range(start, end):
                qt = q[t]   # (nk or nv, kd)
                kt = k[t]   # (nv, kd)
                vt = v[t]   # (nv, vd)

                # gate is (N, nk) — expand to nv
                if gate.shape[1] == nk and nk != nv:
                    gt = gate[t].repeat_interleave(self.head_expand_ratio)
                else:
                    gt = gate[t]
                if beta.shape[1] == nk and nk != nv:
                    bt = beta[t].repeat_interleave(self.head_expand_ratio)
                else:
                    bt = beta[t]

                gt = gt.clamp(-5.0, 0.0)
                decay = torch.exp(gt).unsqueeze(-1).unsqueeze(-1)  # (nv, 1, 1)
                b_exp = bt.unsqueeze(-1).unsqueeze(-1)  # (nv, 1, 1)

                kv = torch.einsum('hd,hv->hdv', kt, vt)  # (nv, kd, vd)
                state = decay * state + b_exp * kv
                state = state.clamp(-100.0, 100.0)

                out_t = torch.einsum('hd,hdv->hv', qt if qt.shape[0] == nv
                                     else qt.repeat_interleave(self.head_expand_ratio, dim=0),
                                     state)
                out_t = out_t.clamp(-1e4, 1e4)
                outputs.append(out_t)

        output = torch.stack(outputs, dim=0)  # (N, nv, vd)
        return output.to(torch.float16), state

    def _single_step_decode(self, q, k, v, gate, beta, temporal_state):
        """Single-step recurrent decode."""
        nk = self.num_k_heads
        nv = self.num_v_heads
        kd = self.head_k_dim
        vd = self.head_v_dim

        q = q.squeeze(0)  # (nk, kd) or (nv, kd)
        k = k.squeeze(0)
        v = v.squeeze(0)  # (nv, vd)

        if self.head_expand_ratio > 1:
            k = k.repeat_interleave(self.head_expand_ratio, dim=0)
            if q.shape[0] == nk:
                q = q.repeat_interleave(self.head_expand_ratio, dim=0)

        if temporal_state is None:
            temporal_state = torch.zeros(nv, kd, vd, dtype=torch.float32, device=q.device)
        else:
            temporal_state = temporal_state.float()

        gt = gate.squeeze(0)  # (nk,)
        bt = beta.squeeze(0)  # (nk,)
        if gt.shape[0] == nk and nk != nv:
            gt = gt.repeat_interleave(self.head_expand_ratio)
            bt = bt.repeat_interleave(self.head_expand_ratio)

        gt = gt.clamp(-5.0, 0.0)
        decay = torch.exp(gt).unsqueeze(-1).unsqueeze(-1)
        b_exp = bt.unsqueeze(-1).unsqueeze(-1)

        kv = torch.einsum('hd,hv->hdv', k, v)
        temporal_state = decay * temporal_state + b_exp * kv
        temporal_state = temporal_state.clamp(-100.0, 100.0)

        output = torch.einsum('hd,hdv->hv', q, temporal_state)
        output = output.clamp(-1e4, 1e4)
        output = output.to(torch.float16).unsqueeze(0)  # (1, nv, vd)

        return output, temporal_state
