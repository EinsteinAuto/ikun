from typing import Union

import ixformer._C as ops
import torch

__all__ = ["vocab_parallel_cross_entropy", "ref_vocab_parallel_cross_entropy"]


def ref_vocab_parallel_cross_entropy(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0,
    world_size: int = 1,
    vocab_start_index: int = 0,
    vocab_end_index: int = 320000,
    group=None,
):
    if world_size == 1:
        vocab_parallel_logits = vocab_parallel_logits.float()
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(
            start=0, end=logits_2d.size()[0], device=logits_2d.device
        )

        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()

        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        loss = torch.log(sum_exp_logits) - predicted_logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        if label_smoothing > 0:
            """
            We'd like to assign 1 / (K - 1) probability mass to every index that is not the ground truth.
            = (1 - alpha) * y_gt + alpha * mean(y_{i for i != gt})
            = (1 - alpha) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = ((K - 1) * (1 - alpha) / (K - 1)) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = (K * (1 - alpha) - 1) / (K - 1)) * y_gt  + (alpha / (K - 1)) * \sum_{i} y_i
            = (1 - (alpha * K) / (K - 1)) * y_gt + ( (alpha * K) / (K - 1) ) * \sum_{i} y_i / K
            From: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/losses/smoothed_cross_entropy.py
            """
            assert 1.0 > label_smoothing > 0.0
            smoothing = label_smoothing * partition_vocab_size / (partition_vocab_size - 1)

            # Exp logits at this point are normalized probabilities. So we can just take the log to get log-probs.
            log_probs = torch.log(exp_logits)
            mean_log_probs = log_probs.mean(dim=-1)
            loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

    else:
        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        torch.distributed.all_reduce(
            logits_max, op=torch.distributed.ReduceOp.MAX, group=group
        )
        # Subtract the maximum value.
        vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)

        # Get the partition's vocab indecies
        partition_vocab_size = vocab_parallel_logits.size()[-1]

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(
            start=0, end=logits_2d.size()[0], device=logits_2d.device
        )
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(
            predicted_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=group,
        )

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(
            sum_exp_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=group,
        )

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Normalize and optionally smooth logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        vocab_size = exp_logits.size(-1)
        if label_smoothing > 0:
            """
            We'd like to assign 1 / (K - 1) probability mass to every index that is not the ground truth.
            = (1 - alpha) * y_gt + alpha * mean(y_{i for i != gt})
            = (1 - alpha) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = ((K - 1) * (1 - alpha) / (K - 1)) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = (K * (1 - alpha) - 1) / (K - 1)) * y_gt  + (alpha / (K - 1)) * \sum_{i} y_i
            = (1 - (alpha * K) / (K - 1)) * y_gt + ( (alpha * K) / (K - 1) ) * \sum_{i} y_i / K
            From: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/losses/smoothed_cross_entropy.py
            """
            assert 1.0 > label_smoothing > 0.0
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)

            # Exp logits at this point are normalized probabilities. So we can just take the log to get log-probs.
            log_probs = torch.log(exp_logits)
            mean_log_probs = log_probs.mean(dim=-1)
            loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

    return loss


def vocab_parallel_cross_entropy(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0,
    world_size: int = 1,
    vocab_start_index: int = 0,
    vocab_end_index: int = 320000,
    group=None,
):
    """
    Args:
        vocab_parallel_logits:              (seq_len,1,vocal_size)              torch.float16, torch.bfloat16, torch.float
        target:                             (seq_len,1)                         torch.int64
        label_smoothing:                                                        float
                默认为0.0. 用于标签平滑
        world_size:                                                             int
                当world_size = 1时,目前只支持batch_size = 1 的情况
        vocab_start_index:                                                      int
        vocab_end_index:                                                        int
        group:  
                TP 并行组                                                  
    Returns:
        loss:                                (seq_len,1)                        torch.float          
    """
    if world_size == 1:
        device = vocab_parallel_logits.device
        xnumel = vocab_parallel_logits.shape[0]
        rnumel = vocab_parallel_logits.shape[-1]
        exp_logits = torch.empty(
            (xnumel, 1, rnumel), device=device, dtype=torch.float32
        )
        masked_target_1d = torch.empty((xnumel,), device=device, dtype=torch.int32)
        loss = torch.empty((xnumel, 1), device=device, dtype=torch.float32)

        ops.train.cross_entropy_loss_forward(
            vocab_parallel_logits, target.int(), exp_logits, masked_target_1d, loss
        )

        vocab_size = exp_logits.size(-1)
        if label_smoothing > 0:
            """
            We'd like to assign 1 / (K - 1) probability mass to every index that is not the ground truth.
            = (1 - alpha) * y_gt + alpha * mean(y_{i for i != gt})
            = (1 - alpha) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = ((K - 1) * (1 - alpha) / (K - 1)) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = (K * (1 - alpha) - 1) / (K - 1)) * y_gt  + (alpha / (K - 1)) * \sum_{i} y_i
            = (1 - (alpha * K) / (K - 1)) * y_gt + ( (alpha * K) / (K - 1) ) * \sum_{i} y_i / K
            From: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/losses/smoothed_cross_entropy.py
            """
            assert 1.0 > label_smoothing > 0.0
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)

            # Exp logits at this point are normalized probabilities. So we can just take the log to get log-probs.
            log_probs = torch.log(exp_logits)
            mean_log_probs = log_probs.mean(dim=-1)
            loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

        # Store softmax, target-mask and masked-target for backward pass.

        return loss
    else:
        loss = ref_vocab_parallel_cross_entropy(
            vocab_parallel_logits,
            target,
            label_smoothing,
            world_size,
            vocab_start_index,
            vocab_end_index,
            group,
        )
        return loss
