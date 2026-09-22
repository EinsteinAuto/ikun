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

import torch
from .utils import (
    fast_dequantize,
    QUANT_STATE,
    get_lora_parameters,
    matmul_lora,
    torch_amp_custom_fwd,
    torch_amp_custom_bwd,
)


class LoRA_MLP(torch.autograd.Function):
    """
    ### LoRA weights
    G = G + Ag @ Bg
    U = U + Au @ Bu
    W = W + Aw @ Bw

    ### SwiGLU(X)
    e = X @ G
    f = e * sigmoid(e)
    g = X @ U
    h = f * g
    i = h @ W

    ### Backpropagation chain rule
    See our blog post for more details

    df = sigmoid(e) * (1 - f) + f
    dC/dW = h.T @ dY
    dC/dU = X.T @ (D @ W.T * f)
    dC/dG = X.T @ (D @ W.T * df * g)

    ### Down projection LoRA weights
    dC/dAw = dC/dW @ B.T
    dC/dBw = A.T @ dC/dW
    dC/dAw =       h.T @ dY @ B.T
    dC/dBw = A.T @ h.T @ dY

    ### Up projection LoRA weights
    dC/dAu =       X.T @ (D @ W.T * f) @ B.T
    dC/dBu = A.T @ X.T @ (D @ W.T * f)

    ### Gate projection LoRA weights
    dC/dAg =       X.T @ (D @ W.T * df * g) @ B.T
    dC/dBg = A.T @ X.T @ (D @ W.T * df * g)

    Don't forget to see our blog post for more details!
    """
    @staticmethod
    @torch_amp_custom_fwd
    def forward(ctx, X : torch.Tensor,
                gateupW, gateupW_quant, gateupA, gateupB, gateupS,
                downW, downW_quant, downA, downB, downS,
                _forward_function, _backward_function,):
        dtype = X.dtype

        res_gateup_proj = matmul_lora(X, gateupW, gateupW_quant, gateupA, gateupB, gateupS)
        res_swiglu = _forward_function(res_gateup_proj)
        res_mlp = matmul_lora(res_swiglu, downW, downW_quant, downA, downB, downS)

        ctx.custom_saved_tensors = (
            gateupW, gateupW_quant, gateupS,
            downW, downW_quant, downS,
            _backward_function,
        )
        ctx.save_for_backward(gateupA, gateupB, downA, downB, X, res_gateup_proj, res_mlp)
        return res_mlp
    pass


    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dY : torch.Tensor):
        gateupW, gateupW_quant, gateupS, downW, downW_quant, downS, \
            _backward_function = ctx.custom_saved_tensors
        gateupA, gateupB, downA, downB, \
            X, res_gateup_proj, res_mlp = ctx.saved_tensors

        gateupA, gateupB, downA, downB = \
            gateupA.t(), gateupB.t(), downA.t(), downB.t()

        batch, seq_len, hd = X.shape
        dY = dY.view(-1, dY.shape[-1])
        X  = X .view(-1, X .shape[-1])
        res_gateup_proj  = res_gateup_proj.view(-1, res_gateup_proj.shape[-1])
        dtype = X.dtype

        D_swiglu = matmul_lora(dY, downW.t(), downW_quant, downB, downA, downS)
        DW, e, g = _backward_function(D_swiglu, res_gateup_proj)
        h, df, de = DW, e, g

        # Down projection LoRA weights
        d_downA = h.t() @ (dY @ downB.t())
        d_downB = (downA.t() @ h.t()) @ dY
        d_downA *= downS
        d_downB *= downS

        # Gate_up projection LoRA weights
        d_gateupA = X.t() @ (de @ gateupB.t())
        d_gateupB = (gateupA.t() @ X.t()) @ de
        d_gateupA *= gateupS
        d_gateupB *= gateupS

        # dX  = matmul_lora(df, upW.t(), upW_quant, upB, upA, upS)
        # dX += matmul_lora(de, gateW.t(), gateW_quant, gateB, gateA, gateS)

        gateupW = fast_dequantize(gateupW.t(), gateupW_quant)
        dX = de @ gateupW.t()
        del gateupW
        dX += de @ gateupB.to(dtype).t() @ (gateupS * gateupA.to(dtype).t())

        # gateW, gateW_quant, gateA, gateB, gateS,
        #  upW,    upW_quant,   upA,   upB,   upS,
        # downW, downW_quant, downA, downB, downS,
        return dX.view(batch, seq_len, hd), \
            None, None, d_gateupA.t(), d_gateupB.t(), None, \
            None, None, d_downA.t(), d_downB.t(), None, \
            None, None, # _backward and _forward
    pass
pass


from .swiglu_ import swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel
def apply_lora_mlp_swiglu(self, X):
    gateupW, gateupW_quant, gateupA, gateupB, gateupS = get_lora_parameters(self.gate_up)
    downW, downW_quant, downA, downB, downS = get_lora_parameters(self.down_proj)

    out = LoRA_MLP.apply(X,
                         gateupW, gateupW_quant, gateupA, gateupB, gateupS,
                         downW, downW_quant, downA, downB, downS,
                         swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel,)
    return out
pass