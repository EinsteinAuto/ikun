import torch
import os
from einops import rearrange
from flash_attn import flash_attn_varlen_func


@staticmethod
def replace_flash_attn_forward(self, q, k, v, attention_mask, query_length, dropout=0.0, softmax_scale=None):

    # flash-attn(ixdnn)存在gpt2(118M,338M,738M) shape没适配，只能采用普通版本
    assert os.getenv('ENABLE_FLASH_ATTENTION_WITH_IXDNN', "1") == '0', "flash-attn should not be use ixdnn version, please set variables" \
        " in shell \"export ENABLE_FLASH_ATTENTION_WITH_IXDNN=0 \" "
    assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q, k, v)))
    assert all((i.is_cuda for i in (q, k, v)))

    batch_size, seqlen_q = q.shape[0], q.shape[1]
    seqlen_k = k.shape[1]

    q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                device=q.device)

    if query_length != 1:
        # during training q,k,v always have same seqlen
        assert seqlen_k == seqlen_q

        is_causal = self.is_causal
        cu_seqlens_k = cu_seqlens_q
        dropout_p = dropout
    else:
        # turn off FA causal mask after first inference autoregressive iteration
        # only on first autoregressive step q,k,v have same seqlen
        is_causal = seqlen_q == seqlen_k
        cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                                    device=q.device)
        dropout_p = 0

    output = flash_attn_varlen_func(
        q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
        dropout_p,
        softmax_scale=softmax_scale, causal=is_causal
    )
    # print(f"{output}")
    output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
    return output
