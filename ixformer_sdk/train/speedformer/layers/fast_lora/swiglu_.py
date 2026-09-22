# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import triton
import triton.language as tl
import torch


@triton.jit
def _fg_kernel(x, h, hd, BLOCK_SIZE : tl.constexpr,):
    block_idx = tl.program_id(0)
    offsets0 = block_idx*2*hd + tl.arange(0, BLOCK_SIZE)
    offsets1 = block_idx*2*hd + hd + tl.arange(0, BLOCK_SIZE)
    mask = offsets0 < hd

    e_row = tl.load(x + offsets0, mask = mask, other = 0).to(tl.float32)
    g_row = tl.load(x + offsets1, mask = mask, other = 0)#.to(tl.float32)

    # f = e * sigmoid(e)
    f_row = e_row * tl.sigmoid(e_row) # e_row / (1 + tl.exp(-e_row))
    f_row = f_row.to(g_row.dtype) # Exact copy from HF
    # h = f * g
    h_row = f_row * g_row

    # Store h
    tl.store(h + offsets0, h_row, mask = mask)
pass


def swiglu_fg_kernel(x):
    batch, seq_len, hdx2 = x.shape
    hd = hdx2 // 2
    n_rows = batch * seq_len
    BLOCK_SIZE = triton.next_power_of_2(hd)
    h = torch.empty((batch, seq_len, hd), dtype = x.dtype, device = "cuda:0")

    _fg_kernel[n_rows,](x, h, hd, BLOCK_SIZE=BLOCK_SIZE)
    return h
pass


@triton.jit
def _DWf_DW_dfg_kernel(DW, x, hd, BLOCK_SIZE : tl.constexpr,):
    """
    e = e.float()
    se = 1.0 / (1.0 + torch.exp(-e))
    f = (se * e).to(dtype)
    h = f * g
    df = DW * f
    dg = DW * g
    de = (dg.float() * se * (1.0 + e * (1.0 - se))).to(dtype)
    """
    block_idx = tl.program_id(0)
    offsets0 = block_idx*hd*2 + tl.arange(0, BLOCK_SIZE)
    offsets1 = block_idx*hd*2 + hd + tl.arange(0, BLOCK_SIZE)
    mask = BLOCK_SIZE < hd

    DW_row = tl.load(DW + offsets0, mask = mask, other = 0)#.to(tl.float32)
    e_row  = tl.load(x  + offsets0, mask = mask, other = 0).to(tl.float32)
    g_row  = tl.load(x  + offsets1, mask = mask, other = 0)#.to(tl.float32)

    # e = e.float()
    # se = 1.0 / (1.0 + torch.exp(-e))
    se_row = tl.sigmoid(e_row) # 1.0 / (1.0 + tl.exp(-e_row))
    # f = (se * e).to(dtype)
    f_row = se_row * e_row
    f_row = f_row.to(DW_row.dtype)
    # h = f * g
    h_row  =  f_row * g_row
    # df = DW * f
    df_row = DW_row * f_row
    # dg = DW * g
    dg_row = DW_row * g_row
    # de = (dg.float() * se * (1.0 + e * (1.0 - se))).to(dtype)
    de_row = dg_row.to(tl.float32) * se_row * (1.0 + e_row * (1.0 - se_row))
    de_row = de_row.to(DW_row.dtype)

    # Store derivatives in buffers
    tl.store(DW + offsets0, h_row,  mask = mask) # h  = f * g
    tl.store(x  + offsets0, df_row, mask = mask) # df = DW * f
    tl.store(x  + offsets1, de_row, mask = mask) # de
pass


def swiglu_DWf_DW_dfg_kernel(DW, x):
    batch_seq_len, hdx2 = x.shape
    hd = hdx2 // 2
    BLOCK_SIZE = triton.next_power_of_2(hd)
    _DWf_DW_dfg_kernel[batch_seq_len, ](DW, x, hd, BLOCK_SIZE=BLOCK_SIZE,)
    return DW, x
pass
