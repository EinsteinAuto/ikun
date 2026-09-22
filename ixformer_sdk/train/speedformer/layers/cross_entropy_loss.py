import time
import numpy as np
import torch
import triton
import triton.language as tl
from packaging.version import Version
if Version(triton.__version__) >= Version("3.0.0"):
    from triton.language.extra import libdevice
    triton_tanh = libdevice.tanh
else:
    import triton.language as tl
    triton_tanh = tl.math.tanh


def calculate_settings(n):
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


@triton.heuristics({"DO_SOFTCAPPING": lambda args: args["DO_SOFTCAPPING"], })
@triton.jit
def _cross_entropy_forward(
    logits_ptr, logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr,
):
    """
        Cross Entropy Loss = 1/n sum [ -yi log(Pi) ]
        Pi = exp(xi) / sum(exp(xi))
        CE_i = -y log(p) = -y log[ exp(x) / sum(exp(x)) ]
             = -y [ x - log[sum(exp(x))] ]
             = y * (log[sum(exp(x))] - x)
        If y == 0: CE_i = 0
        If y == 1: CE_i = logsumexp - x

        logsumexp is also stable
        Take    y =         log[sum(exp(x))]
           exp(y) =             sum(exp(x))
           exp(y) =             sum(exp(x - c)*exp(c)) Since e^(x-c)*e^c = e^x
           exp(y) =      exp(c)*sum(exp(x - c))
               y  = log(exp(c)*sum(exp(x - c)))
               y  = c + log[sum(exp(x - c))]
        This means we can set c = max(x) to make sure
        exp(x - c) always is exp(x - max(x)).
        This ensures exp(x - max(x))'s maximum is 1 as exp(0) = 1.
    """
    row_idx = tl.program_id(0)
    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    loss_ptr += row_idx
    logsumexp_ptr += row_idx
    labels_ptr += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
    # Do logit softcapping for Gemma 2: t * tanh(1/t * x)
    if DO_SOFTCAPPING:
        logits = SOFTCAP * triton_tanh(logits / SOFTCAP)

    logits = logits.to(tl.float32)
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if label_idx != -100:
        x = tl.load(logits_ptr + label_idx)
        # Do logit softcapping for Gemma 2: t * tanh(1/t * x)
        if DO_SOFTCAPPING:
            x = SOFTCAP * triton_tanh(x / SOFTCAP)
        loss = logsumexp - x.to(tl.float32)
    else:
        loss = 0.0
    tl.store(logsumexp_ptr, logsumexp)
    tl.store(loss_ptr, loss)


@triton.heuristics({"DO_SOFTCAPPING": lambda args: args["DO_SOFTCAPPING"], })
@triton.jit
def _chunked_cross_entropy_forward(
    logits_ptr, logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr,
):
    """
        256K vocab divided in 4 chunks

        |-65536-| |-65536-| |-65536-| |-65536-|
        |-------| |-------| |-------| |-------|
        |-------| |-------| |-------| |-------|

        If y == 0: CE_i = 0
        If y == 1: CE_i = logsumexp - x

        Notice we can do logsumexp for each chunk and then
        logsumexp[chunk_sum(logsumexp)] == logsumexp

        chunk_sum = log[chunk_sum(logsumexp)]
                  = log[exp(logsumexp(a)) + ... + exp(logsumexp(z))]
                  = log[exp(log[sum(exp(a))]) + ... + exp(log[sum(exp(z))])]
                  = log[sum(exp(a)) + ... + sum(exp(z))]
                  = logsumexp(x)

        This means we can perform a logsumexp for each chunk, then do a
        final logsumexp reduction!

        Ie do: logsumexp(chunked_logsumexp) - x
    """
    row_idx = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    loss_ptr += row_idx
    logsumexp_ptr += row_idx * N_CHUNKS + chunk_idx
    labels_ptr += row_idx

    col_offsets = chunk_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
    # Do logit softcapping for Gemma 2: t * tanh(1/t * x)
    if DO_SOFTCAPPING:
        logits = SOFTCAP * triton_tanh(logits / SOFTCAP)

    logits = logits.to(tl.float32)
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if chunk_idx == 0:
        # logsumexp(chunked_logsumexp) - x
        # Do the -x separately
        if label_idx != -100:
            x = tl.load(logits_ptr + label_idx).to(tl.float32)
            # Do logit softcapping for Gemma 2: t * tanh(1/t * x)
            if DO_SOFTCAPPING:
                x = SOFTCAP * triton_tanh(x / SOFTCAP)
            loss = -1.0 * x.to(tl.float32)
        else:
            loss = 0.0
        tl.store(loss_ptr, loss)

    tl.store(logsumexp_ptr, logsumexp)


@triton.heuristics({"DO_SOFTCAPPING": lambda args: args["DO_SOFTCAPPING"], })
@triton.jit
def _cross_entropy_backward(
    logits_ptr, logits_row_stride,
    dloss_ptr,   dloss_row_stride,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr,
):
    """
        CE_i = -y log(P) = y * (log[sum(exp(x))] - x)
        dC/dx = d/dx (y * log[sum(exp(x))] - x * y)

        From https://en.wikipedia.org/wiki/LogSumExp
        d/dx logsumexp = exp(x) / sum(exp(x)) = softmax(x)

        dC/dx = y * exp(x) / sum(exp(x)) - d/dx (x * y)
        dC/dx = y * exp[ log[exp(x) / sum(exp(x))] ] using x = exp(log(x)) trick
        dC/dx = y * exp[x - logsumexp] - d/dx (x * y)

        If y == 0: dC/dx = 0
        If y == 1 and x == label: dC/dlabel = exp[x - logsumexp] - 1
        If y == 1 and x != label: dC/dx     = exp[x - logsumexp]
    """
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    dloss_ptr += row_idx * dloss_row_stride
    col_offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int32)

    if label_idx != -100:
        dloss = tl.load(dloss_ptr)
    else:
        dloss = 0.0

    x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
    # Do logit softcapping for Gemma 2: t * tanh(1/t * x)
    if DO_SOFTCAPPING:
        # d/dx [t * tanh(1/t * x)] = 1 - tanh^2(1/t * x)
        partial = triton_tanh(x / SOFTCAP)
        x = SOFTCAP * partial

    logsumexp = tl.load(logsumexp_ptr + row_idx)
    y = tl.exp(x.to(tl.float32) - logsumexp)
    y = tl.where(
        col_offsets == label_idx,
        y - 1.0,  # exp(x - logsumexp) - 1
        y,       # exp(x - logsumexp)
    )

    if DO_SOFTCAPPING:
        # d/dx [t * tanh(1/t * x)] = 1 - tanh^2(1/t * x)
        y = y * (1.0 - partial*partial)

    # If y == 0: dC/dx = 0 ==> we already masked it to be = 0, so dloss = 0.
    tl.store(logits_ptr + col_offsets, dloss * y, mask=mask)


MAX_FUSED_SIZE = 65536  # 2**16


class Fast_CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, logit_softcapping=0):
        n_rows, vocab_size = logits.shape

        div, mod = divmod(vocab_size, MAX_FUSED_SIZE)
        n_chunks = div + (mod != 0)
        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)

        DO_SOFTCAPPING = (logit_softcapping != 0)

        if n_chunks == 1:
            # For small vocabs <= 65336 like Llama, Mistral
            BLOCK_SIZE, num_warps = calculate_settings(vocab_size)
            logsumexp = torch.empty(
                n_rows, dtype=torch.float32, device=logits.device)

            _cross_entropy_forward[(n_rows,)](
                logits, logits.stride(0),
                losses,
                logsumexp,
                labels,
                VOCAB_SIZE=vocab_size,
                BLOCK_SIZE=BLOCK_SIZE,
                DO_SOFTCAPPING=DO_SOFTCAPPING,
                SOFTCAP=logit_softcapping,
                num_warps=num_warps,
            )
        else:
            # For large vocabs > 65336 like Gemma 256K
            logsumexp = torch.empty(
                (n_rows, n_chunks,), dtype=torch.float32, device=logits.device)

            _chunked_cross_entropy_forward[(n_rows, n_chunks,)](
                logits, logits.stride(0),
                losses,
                logsumexp,
                labels,
                VOCAB_SIZE=vocab_size,
                N_CHUNKS=n_chunks,
                BLOCK_SIZE=MAX_FUSED_SIZE,
                DO_SOFTCAPPING=DO_SOFTCAPPING,
                SOFTCAP=logit_softcapping,
                num_warps=32,
            )
            # logsumexp(chunked_logsumexp) - x
            # Do the -x separately
            logsumexp = torch.logsumexp(logsumexp, dim=1)  # Row sum
            losses += logsumexp
            # Don't forget to mask padding out!
            losses.masked_fill_(labels == -100, 0)

        ctx.save_for_backward(logits, logsumexp, labels)
        ctx.DO_SOFTCAPPING = DO_SOFTCAPPING
        ctx.logit_softcapping = logit_softcapping
        return losses

    @staticmethod
    def backward(ctx, dlosses):
        logits, logsumexp, labels = ctx.saved_tensors
        n_rows, vocab_size = logits.shape

        BLOCK_SIZE = 4096
        div, mod = divmod(vocab_size, BLOCK_SIZE)
        n_blocks = div + (mod != 0)

        _cross_entropy_backward[(n_rows, n_blocks,)](
            logits,   logits.stride(0),
            dlosses, dlosses.stride(0),
            logsumexp,
            labels,
            VOCAB_SIZE=vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
            DO_SOFTCAPPING=ctx.DO_SOFTCAPPING,
            SOFTCAP=ctx.logit_softcapping,
            num_warps=8,
        )
        return logits, None, None,


@torch._disable_dynamo
def fast_cross_entropy_loss(logits, labels, logit_softcapping=0):
    """
    Arguments:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len,)
    Returns:
        losses: float
    """
    assert len(logits.size()) == 2 or len(logits.size()) == 3
    if len(logits.size()) == 3:
        batch, seq_len, d = logits.shape
        assert (labels.shape == (batch, seq_len))
        logits = logits.view(batch*seq_len, d)
        labels = labels.view(-1)

    loss = Fast_CrossEntropyLoss.apply(
        logits,
        labels,
        logit_softcapping,
    )
    n_items = torch.count_nonzero(labels != -100)
    return loss.sum() / n_items


if __name__ == "__main__":
    shift_logits_numpy = np.random.randn(4096, 32000).astype(np.float32)
    shift_labels_numpy = np.random.randint(0, 32000, (4096, )).astype(np.int64)

    shift_logits = torch.from_numpy(shift_logits_numpy).cuda()
    shift_labels = torch.from_numpy(shift_labels_numpy).cuda()

    shift_logits_ref = torch.from_numpy(shift_logits_numpy).cuda()
    shift_labels_ref = torch.from_numpy(shift_labels_numpy).cuda()

    shift_logits.requires_grad = True
    shift_logits_ref.requires_grad = True

    # test accuracy
    loss = fast_cross_entropy_loss(shift_logits, shift_labels)
    loss_ref = torch.nn.CrossEntropyLoss()(shift_logits_ref, shift_labels_ref)
    loss.backward()
    loss_ref.backward()

    torch.testing.assert_close(loss, loss_ref)
    torch.testing.assert_close(shift_logits.grad, shift_logits_ref.grad)

    start = time.time()
    for i in range(1000):
        loss = fast_cross_entropy_loss(shift_logits, shift_labels)
        loss.backward()
    print("triton:", time.time() - start)

    start = time.time()
    for i in range(1000):
        loss_ref = torch.nn.CrossEntropyLoss()(shift_logits, shift_labels)
        loss_ref.backward()
    print("torch:", time.time() - start)
