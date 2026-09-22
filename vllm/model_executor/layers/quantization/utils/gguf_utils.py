import torch
import numpy as np
from gguf.constants import GGMLQuantizationType

def get_awq_format(w, group_size=128, w_bit=4):
    org_w_shape = w.shape
    ori_w_dtype = torch.get_default_dtype()
    assert w_bit == 4
    assert w.shape[1] % group_size == 0
    
    in_features = org_w_shape[1]
    w = w.reshape(-1, group_size)
    assert torch.isnan(w).sum() == 0
    
    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2**w_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    w = (
        torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
    ) * scales
    zeros = zeros.view(org_w_shape[0], -1)
    scales = scales.view(org_w_shape[0], -1)
    w = w.reshape(org_w_shape)
    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0
    
    scales = scales.t().contiguous() # input // group, o
    zeros = zeros.t().contiguous()   # input // group, o

    # from auto awq
    scale_zeros = zeros * scales
    scales = scales.clone().to(ori_w_dtype)

    pack_num = 32 // w_bit
    intweight = []
    for idx in range(in_features):
        intweight.append(
            torch.round(
                (w[:, idx] + scale_zeros[idx // group_size])
                / scales[idx // group_size]
            ).to(torch.int)[:, None]
        )
    intweight = torch.cat(intweight, dim=1)
    intweight = intweight.t().contiguous()
    intweight = intweight.to(dtype=torch.int32)

    qweight = torch.zeros(
        (intweight.shape[0], intweight.shape[1] // 32 * w_bit),
        dtype=torch.int32,
        device=intweight.device,
    )

    for col in range(intweight.shape[1] // pack_num):
        order_map = [0, 2, w_bit, 6, 1, 3, 5, 7]
        for i in range(pack_num):
            qweight_col = intweight[:, col * pack_num + order_map[i]]
            qweight[:, col] |= qweight_col << (i * w_bit)

    zeros = zeros.to(dtype=torch.int32, device=qweight.device)

    qzeros = torch.zeros(
        (zeros.shape[0], zeros.shape[1] // 32 * w_bit),
        dtype=torch.int32,
        device=zeros.device,
    )

    for col in range(zeros.shape[1] // pack_num):
        order_map = [0, 2, w_bit, 6, 1, 3, 5, 7]
        for i in range(pack_num):
            qzero_col = zeros[:, col * pack_num + order_map[i]]
            qzeros[:, col] |= qzero_col << (i * w_bit)
    
    return qweight, qzeros, scales

GGML_BLOCK_SIZES = {
    "F32": 4,
    "F16": 2,
    "Q4_0": 2 + 16,
    "Q5_0": 2 + 4 + 16,
    "Q8_0": 2 + 32,
    "Q2_K": 256 // 16 + 256 // 4 + 2 + 2,
    "Q3_K": 256 // 8 + 256 // 4 + 12 + 2,
    "Q4_K": 2 + 2 + 12 + 256 // 2,
    "Q5_K": 2 + 2 + 12 + 256 // 8 + 256 // 2,
    "Q6_K": 256 // 2 + 256 // 4 + 256 // 16 + 2,
    "IQ4_XS": 2 + 2 + 256 // 2 + 256 // 64,
}

def dequantize_f32(data):
    return np.frombuffer(data, dtype=np.float32)

def dequantize_f16(data):
    return np.frombuffer(data, dtype=np.float16)

def dequantize_q4_0(data):
    num_blocks = len(data) // GGML_BLOCK_SIZES["Q4_0"]

    scales = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, 1 + 8)[:, :1].astype(np.float32)
    qs = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, 2 + 16)[:, 2:]

    return np.concatenate([
        scales * ((qs & 0xf).astype(np.int8) - 8),
        scales * ((qs >> 4).astype(np.int8) - 8),
    ], axis=1)

def dequantize_q5_0(data):
    num_blocks = len(data) // GGML_BLOCK_SIZES["Q5_0"]

    scales = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, 1 + 2 + 8)[:, :1].astype(np.float32)
    qh = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, 2 + 4 + 16)[:, 2:2 + 4]
    qs = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, 2 + 4 + 16)[:, 2 + 4:]

    bits = np.unpackbits(qh, axis=-1, bitorder="little")

    x0 = ((qs & 0xf).astype(np.int8) | (bits[:, :16] << 4)) - 16
    x1 = ((qs >> 4).astype(np.int8) | (bits[:, 16:] << 4)) - 16

    return np.concatenate([
        scales * x0,
        scales * x1,
    ], axis=1)
    
def dequantize_q8_0(data):
    num_blocks = len(data) // GGML_BLOCK_SIZES["Q8_0"]

    scales = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, 1 + 16)[:, :1].astype(np.float32)
    qs = np.frombuffer(data, dtype=np.int8).reshape(num_blocks, 2 + 32)[:, 2:]
    return scales * qs

def dequantize_q2_k(data):
    block_size = GGML_BLOCK_SIZES["Q2_K"]
    num_blocks = len(data) // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)

    dmin = data_f16[:, -1].reshape(num_blocks, 1, 1).astype(np.float32)
    d = data_f16[:, -2].reshape(num_blocks, 1, 1).astype(np.float32)
    scales = data_u8[:, :16].reshape(num_blocks, 16, 1)
    qs = data_u8[:, 16:80].reshape(num_blocks, 64)

    tmp = np.stack([
        qs[:, 00:16] >> 0,
        qs[:, 16:32] >> 0,
        qs[:, 00:16] >> 2,
        qs[:, 16:32] >> 2,
        qs[:, 00:16] >> 4,
        qs[:, 16:32] >> 4,
        qs[:, 00:16] >> 6,
        qs[:, 16:32] >> 6,
        qs[:, 32:48] >> 0,
        qs[:, 48:64] >> 0,
        qs[:, 32:48] >> 2,
        qs[:, 48:64] >> 2,
        qs[:, 32:48] >> 4,
        qs[:, 48:64] >> 4,
        qs[:, 32:48] >> 6,
        qs[:, 48:64] >> 6,
    ], axis=1)

    return d * (scales & 15) * (tmp & 3) - dmin * (scales >> 4)


def dequantize_q3_k(data):
    block_size = GGML_BLOCK_SIZES["Q3_K"]
    num_blocks = len(data) // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)

    d = data_f16[:, -1].reshape(num_blocks, 1, 1).astype(np.float32)
    bits = np.unpackbits(data_u8[:, :32].reshape(num_blocks, 32, 1), axis=-1, bitorder="little")
    bits = 4 ^ (bits << 2)
    qs = data_u8[:, 32:32 + 64].astype(np.int16)
    a, b, c = data_u8[:, 96: 96 + 12].reshape(num_blocks, 3, 4).transpose(1, 0, 2)
    scales = np.zeros((num_blocks, 4, 4), dtype=np.uint8)
    scales[:, 0] = (a & 15) | ((c & 3) << 4)
    scales[:, 1] = (b & 15) | (((c >> 2) & 3) << 4)
    scales[:, 2] = (a >> 4) | (((c >> 4) & 3) << 4)
    scales[:, 3] = (b >> 4) | ((c >> 6) << 4)
    scales = scales.reshape(num_blocks, 16, 1).astype(np.int16)

    return d * (scales - 32) * np.stack([
        (((qs[:, 00:16] >> 0) & 3) - bits[:, :16, 0]),
        (((qs[:, 16:32] >> 0) & 3) - bits[:, 16:, 0]),
        (((qs[:, 00:16] >> 2) & 3) - bits[:, :16, 1]),
        (((qs[:, 16:32] >> 2) & 3) - bits[:, 16:, 1]),
        (((qs[:, 00:16] >> 4) & 3) - bits[:, :16, 2]),
        (((qs[:, 16:32] >> 4) & 3) - bits[:, 16:, 2]),
        (((qs[:, 00:16] >> 6) & 3) - bits[:, :16, 3]),
        (((qs[:, 16:32] >> 6) & 3) - bits[:, 16:, 3]),
        (((qs[:, 32:48] >> 0) & 3) - bits[:, :16, 4]),
        (((qs[:, 48:64] >> 0) & 3) - bits[:, 16:, 4]),
        (((qs[:, 32:48] >> 2) & 3) - bits[:, :16, 5]),
        (((qs[:, 48:64] >> 2) & 3) - bits[:, 16:, 5]),
        (((qs[:, 32:48] >> 4) & 3) - bits[:, :16, 6]),
        (((qs[:, 48:64] >> 4) & 3) - bits[:, 16:, 6]),
        (((qs[:, 32:48] >> 6) & 3) - bits[:, :16, 7]),
        (((qs[:, 48:64] >> 6) & 3) - bits[:, 16:, 7])
    ], axis=1)
    
def dequantize_q4_k(data, device=None):
    block_size = GGML_BLOCK_SIZES["Q4_K"]
    num_blocks = len(data) // block_size
    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)
    # Casting to float32 because float16 is very slow on CPU
    scale_factors = data_f16[:, 0].reshape(num_blocks, 1, 1).astype(np.float32)
    scale_offsets = data_f16[:, 1].reshape(num_blocks, 1, 1).astype(np.float32)
    qs1 = data_u8[:, 4:16].reshape(num_blocks, 12, 1)
    qs2 = data_u8[:, 16:].reshape(num_blocks, 4, 32)
    # Dequantize scales and offsets (6 bits and 4 + 2 bits)
    factors = scale_factors * np.concatenate([qs1[:, 0:4] & 0b111111, (qs1[:, 8:] & 15) | ((qs1[:, 0:4] >> 6) << 4)], axis=1)
    offsets = scale_offsets * np.concatenate([qs1[:, 4:8] & 0b111111, (qs1[:, 8:] >> 4) | ((qs1[:, 4:8] >> 6) << 4)], axis=1)
    # Interleave low and high quantized bits
    qs2 = np.stack([qs2 & 0xf, qs2 >> 4], axis=2).reshape(num_blocks, 8, 32)
    # Dequantize final weights using scales and offsets
    weight = factors * qs2 - offsets
    if device is None:
        return weight
    return torch.from_numpy(weight).to(device=device)

def dequantize_q5_k(data):
    block_size = GGML_BLOCK_SIZES["Q5_K"]
    num_blocks = len(data) // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)

    d = data_f16[:, 0].reshape(num_blocks, 1).astype(np.float32)
    dmin = data_f16[:, 1].reshape(num_blocks, 1).astype(np.float32)
    scales = data_u8[:, 4:16].reshape(num_blocks, 12, 1)
    qh = data_u8[:, 16: 16 + 32].reshape(num_blocks, 32, 1)
    qs = data_u8[:, 48: 48 + 128].reshape(num_blocks, 4, 32)

    bits = np.unpackbits(qh, axis=-1, bitorder="little")

    qs_hi_4 = qs >> 4
    qs_lo_4 = qs & 15

    scales_lo_6 = scales[:, :8] & 63
    scales_hi_6 = scales[:, :8] >> 6
    scales_lo_4 = scales[:, 8:] & 15
    scales_hi_4 = scales[:, 8:] >> 4

    m1 = dmin * scales_lo_6[:, 4]
    m2 = dmin * scales_lo_6[:, 5]
    m3 = dmin * scales_lo_6[:, 6]
    m4 = dmin * scales_lo_6[:, 7]
    m5 = dmin * (scales_hi_4[:, 0] | (scales_hi_6[:, 4] << 4))
    m6 = dmin * (scales_hi_4[:, 1] | (scales_hi_6[:, 5] << 4))
    m7 = dmin * (scales_hi_4[:, 2] | (scales_hi_6[:, 6] << 4))
    m8 = dmin * (scales_hi_4[:, 3] | (scales_hi_6[:, 7] << 4))

    d1 = d * scales_lo_6[:, 0]
    d2 = d * scales_lo_6[:, 1]
    d3 = d * scales_lo_6[:, 2]
    d4 = d * scales_lo_6[:, 3]
    d5 = d * (scales_lo_4[:, 0] | (scales_hi_6[:, 0] << 4))
    d6 = d * (scales_lo_4[:, 1] | (scales_hi_6[:, 1] << 4))
    d7 = d * (scales_lo_4[:, 2] | (scales_hi_6[:, 2] << 4))
    d8 = d * (scales_lo_4[:, 3] | (scales_hi_6[:, 3] << 4))

    return np.concatenate([
        d1 * (qs_lo_4[:, 0] + (bits[:, :, 0] << 4)) - m1,
        d2 * (qs_hi_4[:, 0] + (bits[:, :, 1] << 4)) - m2,
        d3 * (qs_lo_4[:, 1] + (bits[:, :, 2] << 4)) - m3,
        d4 * (qs_hi_4[:, 1] + (bits[:, :, 3] << 4)) - m4,
        d5 * (qs_lo_4[:, 2] + (bits[:, :, 4] << 4)) - m5,
        d6 * (qs_hi_4[:, 2] + (bits[:, :, 5] << 4)) - m6,
        d7 * (qs_lo_4[:, 3] + (bits[:, :, 6] << 4)) - m7,
        d8 * (qs_hi_4[:, 3] + (bits[:, :, 7] << 4)) - m8,
    ], axis=1)

def dequantize_q6_k(data, device = None):
    block_size = GGML_BLOCK_SIZES["Q6_K"]
    num_blocks = len(data) // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)
    data_i8 = np.frombuffer(data, dtype=np.int8).reshape(num_blocks, block_size)

    scales = data_f16[:, -1].reshape(num_blocks, 1).astype(np.float32)
    # TODO use uint8 and cast later?
    ql = data_u8[:, :128].astype(np.int16)
    qh = data_u8[:, 128:192].astype(np.int16)
    sc = data_i8[:, 192:208, np.newaxis].astype(np.float32)

    # Unpack bits, subtraction requires signed data type
    q1 = (ql[:,   :32 ] & 0xF) | (((qh[:, :32] >> 0) & 3) << 4) - 32
    q2 = (ql[:, 32:64 ] & 0xF) | (((qh[:, :32] >> 2) & 3) << 4) - 32
    q3 = (ql[:,   :32 ] >>  4) | (((qh[:, :32] >> 4) & 3) << 4) - 32
    q4 = (ql[:, 32:64 ] >>  4) | (((qh[:, :32] >> 6) & 3) << 4) - 32
    q5 = (ql[:, 64:96 ] & 0xF) | (((qh[:, 32:] >> 0) & 3) << 4) - 32
    q6 = (ql[:, 96:128] & 0xF) | (((qh[:, 32:] >> 2) & 3) << 4) - 32
    q7 = (ql[:, 64:96 ] >>  4) | (((qh[:, 32:] >> 4) & 3) << 4) - 32
    q8 = (ql[:, 96:128] >>  4) | (((qh[:, 32:] >> 6) & 3) << 4) - 32

    # Dequantize
    weight = scales * np.concatenate([
        sc[:,  0] * q1[:, :16],
        sc[:,  1] * q1[:, 16:],
        sc[:,  2] * q2[:, :16],
        sc[:,  3] * q2[:, 16:],
        sc[:,  4] * q3[:, :16],
        sc[:,  5] * q3[:, 16:],
        sc[:,  6] * q4[:, :16],
        sc[:,  7] * q4[:, 16:],
        sc[:,  8] * q5[:, :16],
        sc[:,  9] * q5[:, 16:],
        sc[:, 10] * q6[:, :16],
        sc[:, 11] * q6[:, 16:],
        sc[:, 12] * q7[:, :16],
        sc[:, 13] * q7[:, 16:],
        sc[:, 14] * q8[:, :16],
        sc[:, 15] * q8[:, 16:],
    ], axis=1)

    if device is None:
        return weight
    return torch.from_numpy(weight).to(device=device)

QK_K = 256
kvalues_iq4nl = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.int8)

def dequantize_iq4_xs(data):
    block_size = GGML_BLOCK_SIZES["IQ4_XS"]
    num_blocks = len(data) // block_size

    d = np.frombuffer(data, dtype=np.float16)[0::block_size//2].astype(np.float32).reshape(num_blocks, 1)
    scales_h = np.frombuffer(data, dtype=np.uint16)[1::block_size//2].reshape(num_blocks, 1)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)[:, 4:]
    scales_l = data_u8[:, :4].reshape(num_blocks, 4)
    qs = data_u8[:, 4:].reshape(num_blocks, block_size - 8)

    ls = np.zeros((num_blocks, QK_K // 32), dtype=np.int8)
    for ib in range(QK_K // 32):
        ls[:, ib] = ((scales_l[:, ib // 2] >> 4 * (ib % 2)) & 0xf) | (((scales_h[:, 0] >> 2 * ib) & 3) << 4)

    dl = (d * (ls - 32)).reshape(num_blocks, -1, 1)

    qs_lo_4 = qs[:, :QK_K // 2].reshape(num_blocks, -1, 16) & 0xf
    qs_hi_4 = qs[:, :QK_K // 2].reshape(num_blocks, -1, 16) >> 4

    y = np.zeros((num_blocks, QK_K), dtype=np.float32)
    for ib in range(QK_K // 32):
        y[:, ib*32:(ib*32)+16] = dl[:, ib] * kvalues_iq4nl[qs_lo_4[:, ib]]
        y[:, (ib*32)+16:(ib*32)+32] = dl[:, ib] * kvalues_iq4nl[qs_hi_4[:, ib]]

    return y.flatten()

GGML_DEQUANTIZE = {
    int(GGMLQuantizationType.F32): dequantize_f32,
    int(GGMLQuantizationType.F16): dequantize_f16,
    int(GGMLQuantizationType.Q4_0): dequantize_q4_0,
    int(GGMLQuantizationType.Q5_0): dequantize_q5_0,
    int(GGMLQuantizationType.Q8_0): dequantize_q8_0,
    int(GGMLQuantizationType.Q2_K): dequantize_q2_k,
    int(GGMLQuantizationType.Q3_K): dequantize_q3_k,
    int(GGMLQuantizationType.Q4_K): dequantize_q4_k,
    int(GGMLQuantizationType.Q5_K): dequantize_q5_k,
    int(GGMLQuantizationType.Q6_K): dequantize_q6_k,
    int(GGMLQuantizationType.IQ4_XS): dequantize_iq4_xs,
}


def dequant_gguf(data, type, shape):
    values = GGML_DEQUANTIZE[type](data)
    values = torch.from_numpy(values).view(shape)
    return values