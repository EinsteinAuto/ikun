from typing import Union

import ixformer._C as ops
import torch

__all__ = [
    "lightllm_tokenattention",
    "ref_lightllm_tokenattention",
    "lightllm_destindex_copy_kv",
    "ref_lightllm_destindex_copy_kv",
    "lightllm_apply_penalty",
    "ref_lightllm_apply_penalty",
    "lightllm_glm2_rope",
    "ref_lightllm_glm2_rope",
]


def ref_lightllm_glm2_rope(
    x: torch.Tensor,  # tokens,head_num,head_dim
    cos: torch.Tensor,  # tokens,rotdim
    sin: torch.Tensor,
):
    num_tokens, _, rot_dim = list(cos.shape)
    head_num = x.shape[1]
    x12 = x[:, :, : rot_dim * 2]
    x3 = x[:, :, rot_dim * 2 :]
    x12 = x12.reshape(num_tokens, head_num, rot_dim, 2)
    x1 = x12[:, :, :, 0]
    x2 = x12[:, :, :, 1]

    # out0 = q0 * cos - q1 * sin
    # out1 = q0 * sin + q1 * cos
    # q1, q2 是沿着 head_dim维度，交叉取值的

    q1 = x1 * cos - x2 * sin
    q2 = x2 * cos + x1 * sin

    q12 = torch.stack([q1, q2], dim=-1)
    q12 = q12.reshape(num_tokens, head_num, -1)
    x_pytorch = torch.cat([q12, x3], dim=-1)
    return x_pytorch


def lightllm_glm2_rope(
    x: torch.Tensor,  # tokens,head_num,head_dim
    cos: torch.Tensor,  # tokens,rotdim
    sin: torch.Tensor,
):
    """
    Args:
        x:                          (num_tokens, head_num, head_dim)                          torch.half
        cos:                        (num_tokens,1,head_dim//2//2)                             torch.half
        sin:                        (num_tokens,1,head_dim//2//2)                             torch.half
    Returns:
        x:                          (num_tokens, head_num, head_dim)                          torch.half
        
    """
    if isinstance(x, torch.Tensor):
        ops.infer.lightllm_glm2_rope(x, cos, sin)
        return x
    else:
        raise NotImplementedError()


def ref_lightllm_apply_penalty(
    Logits: torch.Tensor,
    presence_penalty: torch.Tensor,
    freqency_penalty: torch.Tensor,
    p_token_ids: torch.Tensor,
    p_token_counts: torch.Tensor,
    p_cumsum_seq_len: torch.Tensor,
    p_max_len_in_batch: int,
):
    batch_size = Logits.size(0)
    output = Logits.clone()
    for cur_batch in range(batch_size):
        cur_freqency = freqency_penalty[cur_batch]
        cur_presence = presence_penalty[cur_batch]
        cur_batch_start_index = p_cumsum_seq_len[cur_batch]
        cur_batch_end_index = p_cumsum_seq_len[cur_batch + 1]
        for token_idx in range(cur_batch_start_index, cur_batch_end_index):
            batch_ids = p_token_ids[token_idx]
            batch_ids_count = p_token_counts[token_idx]
            cur_logits = output[cur_batch][batch_ids]

            freq_logits = cur_logits - batch_ids_count * cur_freqency
            pre_logits = freq_logits - cur_presence
            # if token_idx==0:
            #     print(f"batch_ids {batch_ids} cur_logits {cur_logits} pre_logits {pre_logits}")
            output[cur_batch][batch_ids] = pre_logits
    return output


def lightllm_apply_penalty(
    Logits: torch.Tensor,
    presence_penalty: torch.Tensor,
    freqency_penalty: torch.Tensor,
    p_token_ids: torch.Tensor,
    p_token_counts: torch.Tensor,
    p_cumsum_seq_len: torch.Tensor,
    p_max_len_in_batch: int,
):
    """
    Args:
        logits:                  (batch_size, vocab_size)                       torch.float
        presence_penalty:        (batch_size)                                   torch.float
        freqency_penalty:        (batch_size)                                   torch.float
        p_token_ids:             (num_tokens)                                   torch.int
        p_token_counts:          (num_tokens)                                   torch.int
        p_cumsum_seq_len:        (batch_size+1)                                 torch.int  
        p_max_len_in_batch:                                                     int
                            在一个batch中seq的最大长度
    Returns:
        logits:                  (batch_size, vocab_size)                       torch.float
    """
    if isinstance(Logits, torch.Tensor):
        ops.infer.lightllm_apply_penalty(
            Logits,
            presence_penalty,
            freqency_penalty,
            p_token_ids,
            p_token_counts,
            p_cumsum_seq_len,
            p_max_len_in_batch,
        )
        return Logits
    else:
        raise NotImplementedError()


def ref_lightllm_destindex_copy_kv(
    key_cache: torch.Tensor,
    mem_idx: torch.Tensor,
    output: torch.Tensor,
):
    if key_cache.dim() != 3 or key_cache.size(-1) != 128:
        raise NotImplementedError(
            "lightllm_destindex_copy_kv only support key_cache.dim()==3 and head_size ==128 !"
        )
    output[mem_idx.long()] = key_cache
    return output


def lightllm_destindex_copy_kv(
    key_cache: torch.Tensor,
    mem_idx: torch.Tensor,
    output: torch.Tensor,
):
    """
    Args:
        key_cache:                  (tokens, num_kv_heads,  head_size)                          torch.half
                    目前head_size 只支持128的情况
        mem_idx:                    (tokens)                                                    torch.int
        output:                     (max_tokens, num_kv_heads,  head_size)                      torch.half
    Returns:
        output:                     (max_tokens, num_kv_heads,  head_size)                      torch.half
    """

    if key_cache.dim() != 3 or key_cache.size(-1) != 128:
        raise NotImplementedError(
            "lightllm_destindex_copy_kv only support key_cache.dim()==3 and head_size ==128 !"
        )
    if isinstance(key_cache, torch.Tensor):
        ops.infer.lightllm_destindex_copy_kv(key_cache, mem_idx, output)
    else:
        raise NotImplementedError()
    return output


def ref_lightllm_tokenattention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    reg_tokens: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_seq_len: torch.Tensor,
    scale: float,
    max_context_len: int,
):
    batch_size, tp_q_head_num_, head_dim_ = query.shape
    tp_k_head_num_ = key_cache.size(-2)
    sm_scale = scale
    curbatch_max_context_len = max_context_len
    tmp_k = torch.zeros(
        (batch_size, tp_q_head_num_, curbatch_max_context_len, head_dim_),
        dtype=query.dtype,
        device="cuda",
    )
    tmp_v = torch.zeros(
        (batch_size, tp_q_head_num_, curbatch_max_context_len, head_dim_),
        dtype=query.dtype,
        device="cuda",
    )
    mask = torch.ones([batch_size, 1, 1, curbatch_max_context_len])

    kv_group_num = tp_q_head_num_ // tp_k_head_num_
    for cur_batch in range(batch_size):
        cur_batch_req_idx = b_req_idx[cur_batch]
        seq_len = b_seq_len[cur_batch]
        mask[cur_batch, :, :, :seq_len] = 0
        # print(f"cur_batch {cur_batch}")

        for seq_idx in range(seq_len):
            k_loc = reg_tokens[cur_batch_req_idx][seq_idx]
            # print(k_loc)
            for cur_head in range(tp_q_head_num_):
                cur_kv_head = cur_head // kv_group_num
                tmp_k[cur_batch, cur_head, seq_idx, :] = key_cache[k_loc][cur_kv_head]
                tmp_v[cur_batch, cur_head, seq_idx, :] = value_cache[k_loc][cur_kv_head]
    mask = mask.cuda()
    # batch_size, self.tp_q_head_num_, 1, max_len_in_batch
    attn_score = (
        torch.matmul(
            query.view(batch_size, tp_q_head_num_, 1, head_dim_),
            tmp_k.transpose(-1, -2),
        )
        * sm_scale
    )
    attn_score = attn_score + mask * -1000
    attn_score = torch.softmax(attn_score, dim=-1)
    # batch_size, self.tp_q_head_num_, 1, head_dim
    py_out = torch.matmul(attn_score.to(query.dtype), tmp_v).view(
        batch_size, tp_q_head_num_, -1
    )
    return py_out


def lightllm_tokenattention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    reg_tokens: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_seq_len: torch.Tensor,
    scale: float,
    max_context_len: int,
    partition: int,
    output: torch.Tensor,
):
    """
    Args:
        query:                  (batch_size,head_num,head_dim)                  torch.float16, torch.bfloat16
        key_cache:              (max_num_tokens, head_num_kv, head_dim)         torch.float16, torch.bfloat16
        value_cache:            (max_num_tokens, head_num_kv, head_dim)         torch.float16, torch.bfloat16
        reg_tokens:             (max_request,max_tokens)                        torch.int32
                    目前max_tokens只支持3080
        b_req_idx:              (batch_size)                                    torch.int32
        b_req_len:              (batch_size)                                    torch.int32
        scale:                                                                  float
                    The scaling of QK^T before applying softmax. 
        max_context_len:                                                        int
                    b_seq_len.max()
        partition:                                                              int
    Returns:
        output:                 (batch_size,head_num,head_dim)                  torch.float16, torch.bfloat16
    """
    _,max_tokens=reg_tokens.shape
    if not max_tokens == 3080:
        raise NotImplementedError(
            "lightllm_tokenattention only support reg_tokens.size(-1)==3080"
        )           
    if isinstance(query, torch.Tensor):
        ops.infer.lightllm_tokenattention(
            query,
            key_cache,
            value_cache,
            reg_tokens,
            b_req_idx,
            b_seq_len,
            scale,
            max_context_len,
            partition,
            output,
        )
    else:
        raise NotImplementedError()
    return output
