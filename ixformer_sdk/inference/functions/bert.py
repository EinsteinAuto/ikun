import ixformer._C as ops
import torch

__all__ = [
    "ref_bert_embedding",
    "bert_embedding",
    "ref_bert_add_norm",
    "bert_add_norm",
    "ref_bert_unpack_start_end_logits",
    "bert_unpack_start_end_logits",
    "ref_bert_linear_residual",
    "bert_linear_residual",
]


def ref_bert_embedding(
    token_weight: torch.Tensor,
    pos_weight: torch.Tensor,
    type_weight: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    token_ids: torch.Tensor,
    pos_ids: torch.Tensor,
    type_ids: torch.Tensor,
    epsilon: float = 1e-5,
    out: torch.Tensor = None,
):
    assert out is None
    emd1 = torch.nn.functional.embedding(token_ids, token_weight)
    emd2 = torch.nn.functional.embedding(pos_ids, pos_weight)
    emd3 = torch.nn.functional.embedding(type_ids, type_weight)

    out = emd1 + emd2 + emd3
    out = torch.nn.functional.layer_norm(out, [out.shape[-1]], ln_weight, ln_bias)
    return out


def bert_embedding(
    token_weight: torch.Tensor,
    pos_weight: torch.Tensor,
    type_weight: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    token_ids: torch.Tensor,
    pos_ids: torch.Tensor,
    type_ids: torch.Tensor,
    epsilon: float = 1e-5,
    out: torch.Tensor = None,
):
    """
    Args:
        token_weight:       (vocab_size, hidden_size)    torch.float16, torch.bfloat16
        pos_weight:         (pos_size, hidden_size)      same as token_weight
        type_weight:        (type_size, hidden_size)     same as token_weight
        ln_weight:          (hidden_size)                same as token_weight
        ln_bias:            (hidden_size)                same as token_weight
        token_ids:          (num_tokens)                 torch.int32, torch.int64
        pos_ids:            (num_tokens)                 same as token_ids
        type_ids:           (num_tokens)                 same as token_ids
        epsilon:                                         float
        out:                (num_tokens, hidden_size)    same as token_weight
    Returns:
        out:                (num_tokens, hidden_size)    same as token_weight
    """
    if out is None:
        out_shape = list(token_ids.shape)
        hidden_size = token_weight.shape[-1]
        out_shape.append(hidden_size)
        out = token_weight.new_empty(out_shape)

    ops.infer.bert_embedding(
        token_weight,
        pos_weight,
        type_weight,
        ln_weight,
        ln_bias,
        token_ids,
        pos_ids,
        type_ids,
        out,
        epsilon,
    )

    return out


def ref_bert_add_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    epsilon: float = 1e-5,
    out: torch.Tensor = None,
):
    assert out is None
    input = input + residual
    return torch.nn.functional.layer_norm(
        input, [input.shape[-1]], ln_weight, ln_bias, epsilon
    )


def bert_add_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    epsilon: float = 1e-5,
    out: torch.Tensor = None,
):
    """
    out = input + residual
    out = add_norm(out, ln_weight, ln_bias, epsilon)
    Args:
        input:       (num_tokens, hidden_size)    torch.float16, torch.bfloat16
        residual:    (num_tokens, hidden_size)    same as input
        ln_weight:   (hidden_size)                same as input
        ln_bias:     (hidden_size)                same as input
        epsilon:                                  float
        out:         (num_tokens, hidden_size)    same as input
    Returns:
        out:         (num_tokens, hidden_size)    same as input
    """
    if out is None:
        out = torch.empty_like(input)
    ops.infer.bert_add_norm(input, residual, ln_weight, ln_bias, out, epsilon)
    return out


def ref_bert_unpack_start_end_logits(
    logits: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    max_seq_len: int,
    start_logits: torch.Tensor = None,
    end_logits: torch.Tensor = None,
):
    batch_size = cu_seq_lens.shape[0] - 1
    if start_logits is None:
        start_logits = logits.new_empty([batch_size, max_seq_len])
    if end_logits is None:
        end_logits = logits.new_empty([batch_size, max_seq_len])
    cu_seq_len_cpu = cu_seq_lens.detach().cpu()
    for i in range(batch_size):
        start_idx = cu_seq_len_cpu[i]
        end_idx = cu_seq_len_cpu[i + 1]
        cur_len = end_idx - start_idx
        start_logits[i, :cur_len] = logits[start_idx:end_idx, 0]
        end_logits[i, :cur_len] = logits[start_idx:end_idx, 1]
    return start_logits, end_logits


def bert_unpack_start_end_logits(
    logits: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    max_seq_len: int,
    start_logits: torch.Tensor = None,
    end_logits: torch.Tensor = None,
):
    """
    Args:
        logits:        (num_tokens, 2)              torch.float16, torch.bfloat16
        cu_seq_lens:    (batch_size+1)               torch.int32, torch.int64
        max_seq_len:                                int
        start_logits:  (batch_size, max_seq_len)    same as logits
        end_logits:    (batch_size, max_seq_len)    same as logits
    Returns:
        start_logits:  (batch_size, max_seq_len)    same as logits
        end_logits:    (batch_size, max_seq_len)    same as logits
    """
    batch_size = cu_seq_lens.shape[0] - 1
    if start_logits is None:
        start_logits = logits.new_empty([batch_size, max_seq_len])
    if end_logits is None:
        end_logits = logits.new_empty([batch_size, max_seq_len])
    ops.infer.bert_unpack_start_end_logits(
        logits, cu_seq_lens, start_logits, end_logits
    )
    return start_logits, end_logits


def ref_bert_linear_residual(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, out: torch.Tensor
):
    return torch.nn.functional.linear(input, weight, bias) + out


def bert_linear_residual(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, out: torch.Tensor
):
    """
    Args:
        input:     (m, k)              torch.float16, torch.bfloat16
        weight:    (n, k)              same as input
        bias:      (n)                 same as input
        out:       (m, n)              same as input
    Returns:
        out:       (m, n)              same as input
    """
    ops.infer.bert_linear_residual(input, weight, bias, out)
    return out
