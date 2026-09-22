import torch
import torch.nn as nn
import torch.nn.functional as F

import ixformer.functions as ixf_F


def time_embed(t_emb, weight1, bias1, weight2, bias2):
    # unet time_emd
    # linear + silu + linear
    emb = ixf_F.act_bias_mm(
        t_emb, weight1, act_type="silu", bias=bias1, scale=1, trans_format="TN"
    )
    emb = ixf_F.act_bias_mm(
        emb, weight2, act_type="none", bias=bias2, scale=1, trans_format="TN"
    )
    return emb


def ixf_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    return ixf_F.layernorm(input, weight, bias, normalized_shape)


def ixf_pt_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
):
    if (
        not query.is_contiguous()
        and query.transpose(1, 2).is_contiguous()
        and key.transpose(1, 2).is_contiguous()
        and value.transpose(1, 2).is_contiguous()
        and attn_mask is None
    ):

        batch_size, head_num, seq_len_q, head_dim = query.shape
        _, _, seq_len_k, _ = key.shape

        query = query.transpose(1, 2).view(batch_size * seq_len_q, head_num, head_dim)
        key = key.transpose(1, 2).view(batch_size * seq_len_k, head_num, head_dim)
        value = value.transpose(1, 2).view(batch_size * seq_len_k, head_num, head_dim)

        cu_seqlens_q = torch.arange(
            0,
            seq_len_q * (batch_size + 1),
            seq_len_q,
            dtype=torch.int32,
            device=query.device,
        )
        if seq_len_q == seq_len_k:
            cu_seqlens_k = cu_seqlens_q
        else:
            cu_seqlens_k = torch.arange(
                0,
                seq_len_k * (batch_size + 1),
                seq_len_k,
                dtype=torch.int32,
                device=query.device,
            )

        res = ixf_F.flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q.int(),
            cu_seqlens_k.int(),
            seq_len_q,
            seq_len_k,
        )
        res = res.view(batch_size, seq_len_q, head_num, head_dim).transpose(1, 2)
        return res

    if not query.is_contiguous():
        query = query.contiguous()
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    return ixf_F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, is_causal=is_causal
    )


class UnetIxformerFunction:
    def __init__(self) -> None:
        self.ixf_linear = ixf_F.linear
        self.pt_linear = F.linear
        self.pt_layer_norm = F.layer_norm
        self.pt_scaled_dot_product_attention = F.scaled_dot_product_attention

    def __enter__(self):
        F.linear = self.ixf_linear
        F.layer_norm = ixf_layer_norm
        F.scaled_dot_product_attention = ixf_pt_scaled_dot_product_attention
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        F.linear = self.pt_linear
        F.layer_norm = self.pt_layer_norm
        F.scaled_dot_product_attention = self.pt_scaled_dot_product_attention
        if exc_tb is not None:
            print(f"{exc_type} {exc_val}")
            return False
        return True


def ForwardWrapper(fun):
    def wrap(*args, **kwargs):
        with UnetIxformerFunction() as w:
            return fun(*args, **kwargs)

    return wrap


class IxformerComfyWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.is_ixf_wrapper = True


class Conv2dNhwcWrapper(IxformerComfyWrapper):
    def __init__(self, module):
        super().__init__()
        module.weight.data = module.weight.permute(0, 2, 3, 1).contiguous()
        module.bias.data = module.bias.float()
        self.weight = module.weight.data
        self.bias = module.bias.data
        self.stride = module.stride
        self.padding = module.padding
        self.dilation = module.dilation
        self.groups = module.groups

    def forward(self, x):
        h2 = ixf_F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return h2


class ResBlockNhwcWrapper(IxformerComfyWrapper):
    def __init__(self, module) -> None:
        super().__init__()
        assert not module.updown
        assert not module.use_scale_shift_norm
        assert not module.skip_t_emb
        assert not module.exchange_temb_dims

        if isinstance(module.skip_connection, nn.Identity):
            self.skip_connection = module.skip_connection
        elif get_class_name(module.skip_connection) == "Conv2d":
            self.skip_connection = Conv2dNhwcWrapper(module.skip_connection)
        else:
            raise NotImplementedError(
                f"ResBlockNhwcWrapper support Conv2d or nn.Identity, but got {module.skip_connection}"
            )

        self.in_layers = module.in_layers
        self.out_layers = module.out_layers
        self.emb_layers = module.emb_layers
        self.in_layers_conv = Conv2dNhwcWrapper(module.in_layers[2])
        self.out_layers_conv = Conv2dNhwcWrapper(module.out_layers[3])

    def forward(self, x, emb):
        # x: nhwc
        x1 = x
        # print(x1.shape)
        # fused group_norm silu
        h = ixf_F.group_norm(
            x1,  # nchw->nhwc
            self.in_layers[0].num_groups,
            self.in_layers[0].weight,
            self.in_layers[0].bias,
            format=False,
            act_type=1,
        )
        h = self.in_layers_conv(h)

        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        h = h + emb_out.permute(0, 2, 3, 1)
        # print(h.shape)
        h = ixf_F.group_norm(
            h,
            self.out_layers[0].num_groups,
            self.out_layers[0].weight,
            self.out_layers[0].bias,
            format=False,
            act_type=1,
        )

        h = self.out_layers[2](h)
        h = self.out_layers_conv(h)
        # TODO: support other skip_connection
        return self.skip_connection(x) + h


class DownsampleNhwcWrapper(IxformerComfyWrapper):
    def __init__(self, module) -> None:
        # TODO: support avg_pool_nd
        super().__init__()
        assert module.use_conv
        self.channels = module.channels
        self.op = Conv2dNhwcWrapper(module.op)

    def forward(self, x):
        assert x.shape[-1] == self.channels
        return self.op(x)


def ffn_forward(self, x):
    if get_class_name(self.net[0]) == "GEGLU":
        net = self.net[1:]
        geglu_net = self.net[0]
        x = geglu_net.proj(x)
        x = ixf_F.gelu_and_mul(x)
        return net(x)
    else:
        return self.net(x)


# ComfyUI/comfy/ldm/modules/attention.py `class BasicTransformerBlock(nn.Module)`
def transformer_block_forward(self, x, context=None, transformer_options={}):
    extra_options = {}
    block = transformer_options.get("block", None)
    block_index = transformer_options.get("block_index", 0)
    transformer_patches = {}
    transformer_patches_replace = {}

    for k in transformer_options:
        if k == "patches":
            transformer_patches = transformer_options[k]
        elif k == "patches_replace":
            transformer_patches_replace = transformer_options[k]
        else:
            extra_options[k] = transformer_options[k]

    extra_options["n_heads"] = self.n_heads
    extra_options["dim_head"] = self.d_head

    if self.ff_in:
        x_skip = x
        x = self.ff_in(self.norm_in(x))
        if self.is_res:
            x += x_skip

    n = self.norm1(x)
    if self.disable_self_attn:
        context_attn1 = context
    else:
        context_attn1 = None
    value_attn1 = None

    if "attn1_patch" in transformer_patches:
        patch = transformer_patches["attn1_patch"]
        if context_attn1 is None:
            context_attn1 = n
        value_attn1 = context_attn1
        for p in patch:
            n, context_attn1, value_attn1 = p(
                n, context_attn1, value_attn1, extra_options
            )

    if block is not None:
        transformer_block = (block[0], block[1], block_index)
    else:
        transformer_block = None

    attn1_replace_patch = transformer_patches_replace.get("attn1", {})
    block_attn1 = transformer_block
    if block_attn1 not in attn1_replace_patch:
        block_attn1 = block

    if block_attn1 in attn1_replace_patch:
        if context_attn1 is None:
            context_attn1 = n
            value_attn1 = n
        n = self.attn1.to_q(n)
        context_attn1 = self.attn1.to_k(context_attn1)
        value_attn1 = self.attn1.to_v(value_attn1)
        n = attn1_replace_patch[block_attn1](
            n, context_attn1, value_attn1, extra_options
        )
        n = self.attn1.to_out(n)
    else:
        n = self.attn1(n, context=context_attn1, value=value_attn1)

    if "attn1_output_patch" in transformer_patches:
        patch = transformer_patches["attn1_output_patch"]
        for p in patch:
            n = p(n, extra_options)

    x += n
    if "middle_patch" in transformer_patches:
        patch = transformer_patches["middle_patch"]
        for p in patch:
            x = p(x, extra_options)

    if self.attn2 is not None:
        n = self.norm2(x)
        if self.switch_temporal_ca_to_sa:
            context_attn2 = n
        else:
            context_attn2 = context
        value_attn2 = None
        if "attn2_patch" in transformer_patches:
            patch = transformer_patches["attn2_patch"]
            value_attn2 = context_attn2
            for p in patch:
                n, context_attn2, value_attn2 = p(
                    n, context_attn2, value_attn2, extra_options
                )

        attn2_replace_patch = transformer_patches_replace.get("attn2", {})
        block_attn2 = transformer_block
        if block_attn2 not in attn2_replace_patch:
            block_attn2 = block

        if block_attn2 in attn2_replace_patch:
            if value_attn2 is None:
                value_attn2 = context_attn2
            n = self.attn2.to_q(n)
            context_attn2 = self.attn2.to_k(context_attn2)
            value_attn2 = self.attn2.to_v(value_attn2)
            n = attn2_replace_patch[block_attn2](
                n, context_attn2, value_attn2, extra_options
            )
            n = self.attn2.to_out(n)
        else:
            n = self.attn2(n, context=context_attn2, value=value_attn2)

    if "attn2_output_patch" in transformer_patches:
        patch = transformer_patches["attn2_output_patch"]
        for p in patch:
            n = p(n, extra_options)

    # x += n
    # if self.is_res:
    #     x_skip = x
    # x = self.ff(self.norm3(x))

    x, x_skip = ixf_F.residual_layer_norm(
        n,
        self.norm3.normalized_shape,
        self.norm3.weight,
        self.norm3.bias,
        x,
        eps=self.norm3.eps,
        is_post_ln=False,
    )
    x = ffn_forward(self.ff, x)

    # x = ffn_forward(self.ff, self.norm3(x))
    if self.is_res:
        x += x_skip

    return x


class SpatialTransformerNhwcWrapper(IxformerComfyWrapper):
    def __init__(self, module):
        super().__init__()
        self.use_linear = module.use_linear
        self.transformer_blocks = module.transformer_blocks
        self.norm = module.norm
        if not self.use_linear:
            self.proj_in = Conv2dNhwcWrapper(module.proj_in)
            self.proj_out = Conv2dNhwcWrapper(module.proj_out)
        else:
            self.proj_in = module.proj_in
            self.proj_out = module.proj_out

    @ForwardWrapper
    def forward(self, x, context=None, transformer_options={}):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context] * len(self.transformer_blocks)

        b, h, w, c = x.shape
        x_in = x

        # group_norm
        x = ixf_F.group_norm(
            x,
            self.norm.num_groups,
            self.norm.weight,
            self.norm.bias,
            format=False,
        )
        # conv2d
        if not self.use_linear:
            x = self.proj_in(x)
        # n,(hw),c
        x = x.view(x.shape[0], -1, x.shape[-1])
        if self.use_linear:
            x = self.proj_in(x)

        for i, block in enumerate(self.transformer_blocks):
            transformer_options["block_index"] = i
            # x = block(x, context=context[i], transformer_options=transformer_options)
            x = transformer_block_forward(
                block, x, context=context[i], transformer_options=transformer_options
            )

        if self.use_linear:
            x = self.proj_out(x)
        x = x.view(b, h, w, c)
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class UpsampleNhwcWrapper(IxformerComfyWrapper):
    def __init__(self, module) -> None:
        # TODO: support mhwc interpolate
        super().__init__()
        self.dims = module.dims
        self.use_conv = module.use_conv
        self.channels = module.channels
        if self.use_conv:
            self.conv = Conv2dNhwcWrapper(module.conv)

    def forward(self, x, output_shape=None):
        # print("================== Upsample is running ==================")
        assert x.shape[-1] == self.channels
        assert len(x.shape) == 4

        # nhwc -> nchw
        if output_shape is not None:
            assert len(output_shape) == 4
            output_shape = [
                output_shape[0],
                output_shape[3],
                output_shape[1],
                output_shape[2],
            ]
        x = x.permute(0, 3, 1, 2).contiguous()
        if self.dims == 3:
            shape = [x.shape[2], x.shape[3] * 2, x.shape[4] * 2]
            if output_shape is not None:
                shape[1] = output_shape[3]
                shape[2] = output_shape[4]
        else:
            shape = [x.shape[2] * 2, x.shape[3] * 2]
            if output_shape is not None:
                shape[0] = output_shape[2]
                shape[1] = output_shape[3]
        # TODO: interpolate 支持 nhwc， 去掉前后转置
        x = F.interpolate(x, size=shape, mode="nearest")
        # nchw -> nhwc
        x = x.permute(0, 2, 3, 1).contiguous()
        if self.use_conv:
            x = self.conv(x)
        return x


unet_wrappers = {
    "Conv2d": Conv2dNhwcWrapper,
    "ResBlock": ResBlockNhwcWrapper,
    "Downsample": DownsampleNhwcWrapper,
    "SpatialTransformer": SpatialTransformerNhwcWrapper,
    "Upsample": UpsampleNhwcWrapper,
}


def get_class_name(module):
    return module.__class__.__name__


def module_wrapper(module):
    # 将原始的 module 封装为 nhwc 模式
    module_name = get_class_name(module)
    assert (
        module_name == "TimestepEmbedSequential"
    ), f"ixformer unet_model_wrapper only support 'TimestepEmbedSequential' now, but got {module_name}"

    num_sequential = len(module)
    for idx_seq in range(num_sequential):
        sub_module = module[idx_seq]
        sub_module_name = get_class_name(sub_module)
        # 判断模块是否已经封装
        if not getattr(sub_module, "is_ixf_wrapper", False):
            if sub_module_name in unet_wrappers:
                module[idx_seq].forward = unet_wrappers[sub_module_name](
                    sub_module
                ).forward
                module[idx_seq].is_ixf_wrapper = True
            else:
                raise NotImplementedError(f"{sub_module_name} not support")
    return module
