# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Generator, Tuple, Callable

import gguf
import torch
import torch.nn.functional as F
from gguf import GGMLQuantizationType as WeightType
from torch.nn.parameter import Parameter, UninitializedParameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.layer import (FusedMoE,
                                                        FusedMoEMethodBase)
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, FusedMoEMethodBase, FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.quantization.utils.gguf_utils import get_awq_format, dequant_gguf
import gguf
import ctypes
import ixformer.inference.functions as IXF
try:
    import ktransformers.cpuinfer_ext as cpuinfer_ext
    from ktransformers.cpuinfer_ext.moe import MOEConfig, MOE
except:
    try:
        print("ktransformers may not installed, try import cpuinfer_ext from vllm.")
        import vllm.cpuinfer_ext as cpuinfer_ext
        from vllm.cpuinfer_ext.moe import MOEConfig, MOE
    except:
        print("ktransformers cpu inference component cpuinfer_ext.so not founded. deepseek will not work.")
        cpuinfer_ext = None
        MOEConfig = None
        MOE = None

logger = init_logger(__name__)


class GGUFConfig(QuantizationConfig):
    """Config class for GGUF."""

    def __init__(self, ) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return ("GGUFConfig()")

    def get_name(self) -> str:
        return "gguf"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []  # no extra configs.

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GGUFConfig":
        return cls()

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            return GGUFLinearMethod(self)
        elif isinstance(layer, VocabParallelEmbedding):
            return GGUFEmbeddingMethod(self)
        elif isinstance(layer, FusedMoE):
            return GGUFMoEMethod(self, prefix)
        return None


UNQUANTIZED_TYPES = {WeightType.F32, WeightType.F16, WeightType.BF16}
STANDARD_QUANT_TYPES = {
    WeightType.Q4_0,
    WeightType.Q4_1,
    WeightType.Q5_0,
    WeightType.Q5_1,
    WeightType.Q8_0,
    WeightType.Q8_1,
}
KQUANT_TYPES = {
    WeightType.Q2_K,
    WeightType.Q3_K,
    WeightType.Q4_K,
    WeightType.Q5_K,
    WeightType.Q6_K,
}
IMATRIX_QUANT_TYPES = {
    WeightType.IQ1_M,
    WeightType.IQ1_S,
    WeightType.IQ2_XXS,
    WeightType.IQ2_XS,
    WeightType.IQ2_S,
    WeightType.IQ3_XXS,
    WeightType.IQ3_S,
    WeightType.IQ4_XS,
    WeightType.IQ4_NL,
}
# TODO(Isotr0py): Currently, we don't have MMQ kernel for I-Matrix quantization.
# Consolidate DEQUANT_TYPES, MMVQ_QUANT_TYPES and MMQ_QUANT_TYPES after we add
# MMQ kernel for I-Matrix quantization.
DEQUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMVQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES


def _fuse_mul_mat(x: torch.Tensor, qweight: torch.Tensor,
                  qweight_type: int) -> torch.Tensor:
    # HACK: when doing chunked prefill we don't generate output tokens
    # so input to logits generator is empty which causes invalid parameter
    if x.shape[0] == 0:
        return torch.empty(x.shape[0],
                           qweight.shape[0],
                           dtype=x.dtype,
                           device=x.device)
    # there is no need to call any kernel for fp16/bf16
    if qweight_type in UNQUANTIZED_TYPES:
        return x @ qweight.T
    # enable MMVQ in contiguous batching with batch_size=1
    if x.shape[0] == 1 and qweight_type in MMVQ_QUANT_TYPES:
        y = ops.ggml_mul_mat_vec_a8(qweight, x, qweight_type, qweight.shape[0])
    # Use MMQ Kernel if it's available (standard + k-quants)
    elif qweight_type in MMQ_QUANT_TYPES:
        y = ops.ggml_mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])
    # If there is no available MMQ kernel, fallback to dequantize
    elif qweight_type in DEQUANT_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
        shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
        weight = ops.ggml_dequantize(qweight, qweight_type, *shape, x.dtype)
        y = x @ weight.T
    else:
        # Raise an error if the quantization type is not supported.
        # Might be useful if llama.cpp adds a new quantization type.
        # Wrap to GGMLQuantizationType IntEnum to make sure it's a valid type.
        qweight_type = WeightType(qweight_type)
        raise NotImplementedError(
            f"Unsupported GGUF quantization type: {qweight_type}")
    return y


def _fused_moe_gguf(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    act,
) -> torch.Tensor:
    # lazy import to avoid triggering triton import in CPU backend
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        moe_align_block_size)

    out_hidden_states = torch.empty_like(x)
    if qweight_type2 in MMQ_QUANT_TYPES and qweight_type in MMQ_QUANT_TYPES:
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]
        BLOCK_SIZE = ops.ggml_moe_get_block_size(qweight_type)

        sorted_token_ids, expert_ids, num_tokens_post_padded = \
                moe_align_block_size(topk_ids, BLOCK_SIZE, E)
        out = ops.ggml_moe_a8(x, w1, sorted_token_ids, expert_ids,
                              num_tokens_post_padded, qweight_type, N, top_k,
                              num_tokens)
        out = act(out)
        out = ops.ggml_moe_a8(out, w2, sorted_token_ids, expert_ids,
                              num_tokens_post_padded, qweight_type2,
                              w2.shape[1], 1, num_tokens * top_k)
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1))
        ops.moe_sum(out, out_hidden_states)
    else:
        logger.warning_once("There is no support for fast MoE kernel "
                            "for current quantization method. "
                            "Falling back to slow implementation. ")
        for tok, (w, idx) in enumerate(zip(topk_weights, topk_ids)):
            inp = x[tok].reshape((1, ) + x.shape[1:])
            current_hidden_state = None
            for ww, ii in zip(w, idx):
                expert_up = w1[ii]

                out = _fuse_mul_mat(inp, expert_up, qweight_type)
                out = act(out)

                expert_down = w2[ii]
                current_state = _fuse_mul_mat(out, expert_down,
                                              qweight_type2).mul_(ww)
                if current_hidden_state is None:
                    current_hidden_state = current_state
                else:
                    current_hidden_state.add_(current_state)
            out_hidden_states[tok] = current_hidden_state
    return out_hidden_states


class GGUFLinearMethod(LinearMethodBase):
    """Linear method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            qweight, {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
                "params_dtype": params_dtype,
                "input_size_per_partition" :input_size_per_partition, # restore shape for qkv and merge
                "output_partition_sizes" :output_partition_sizes,
            })
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        qweight_type = Parameter(torch.empty(len(output_partition_sizes),
                                             dtype=torch.uint8),
                                 requires_grad=False)
        set_weight_attrs(
            qweight_type, {
                "is_gguf_weight_type": True,
                "weight_type": 0,
                "shard_weight_type": {},
                "ignore_warning": True
            })
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        shard_id = getattr(layer.qweight, "shard_id", None)
        keep_half_weight = getattr(self.quant_config,"keep_half_weight",None) # use for MLA fused..
        params_dtype = getattr(layer.qweight, "params_dtype", torch.get_default_dtype())
        if shard_id:
            # dequantize shard weights respectively
            shard_id = ["q", "k", "v"] if "q" in shard_id else shard_id
            qweight = layer.qweight.unbind(0)
            weights = []
            input_size_per_partition = layer.qweight.input_size_per_partition
            output_partition_sizes = layer.qweight.output_partition_sizes
            orig_type_weights = []
            for idx in shard_id:
                q_idx = layer.qweight.shard_id_map[idx]
                qweight_type = layer.qweight_type.shard_weight_type[idx]
                weight = dequant_gguf(qweight[q_idx].data.cpu().numpy().reshape(-1), qweight_type, (output_partition_sizes[q_idx], input_size_per_partition))
                weights.append(weight.to(qweight[q_idx].data.device))

            weights = torch.cat(weights, axis=0)
            qweight, qzeros, scales = get_awq_format(weights)
            if keep_half_weight:
                layer.weight = torch.nn.Parameter(weights.to(params_dtype),
                                                  requires_grad=False)
                assert layer.weight.is_cuda
        else:
            qweight = layer.qweight
            qweight_type = layer.qweight_type.weight_type
            weight = dequant_gguf(qweight.data.cpu().numpy().reshape(-1), qweight_type, qweight.tensor_shape)
            qweight, qzeros, scales = get_awq_format(weight.to(qweight.data.device))
            if keep_half_weight:
                layer.weight = torch.nn.Parameter(weight.to(qweight.data.device).to(params_dtype),
                                                  requires_grad=False)
                assert layer.weight.is_cuda
        del layer.qweight
        del layer.qweight_type
        layer.qweight = torch.nn.Parameter(qweight.data,
                                           requires_grad=False)
        layer.qzeros = torch.nn.Parameter(qzeros.data,
                                          requires_grad=False)
        layer.scales = torch.nn.Parameter(scales.data,
                                          requires_grad=False)
        assert layer.qweight.is_cuda
        assert layer.qzeros.is_cuda
        assert layer.scales.is_cuda
        # if layer.weight is not None, we will use it on triton_mla.process_weights_after_loading and delete it.
        # Mainly used to avoid accuracy loss in quantization and dequantization on MLA martix absorption.
        # PS: layer.weight is dequantizing from gguf format
    
    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ gguf format use awq kernel for now..
        shard_id = getattr(layer.qweight, "shard_id", None)

        if shard_id:
            # dequantize shard weights respectively
            shard_id = ["q", "k", "v"] if "q" in shard_id else shard_id
            qweight = layer.qweight.unbind(0)
            result = []
            for idx in shard_id:
                q_idx = layer.qweight.shard_id_map[idx]
                qweight_type = layer.qweight_type.shard_weight_type[idx]
                result.append(_fuse_mul_mat(x, qweight[q_idx], qweight_type))
            out = torch.cat(result, axis=1)
        else:
            qweight = layer.qweight
            qweight_type = layer.qweight_type.weight_type
            out = _fuse_mul_mat(x, qweight, qweight_type)
        if bias is not None:
            out.add_(bias)
        return out
        """
        pack_factor = 8
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        out_shape = (x.shape[:-1] + (qweight.shape[-1] * pack_factor, ))
        reshaped_x = x.reshape(-1, x.shape[-1])
        
        out = ops.awq_gemm(reshaped_x, qweight, scales, qzeros, pack_factor, group_size=128)
        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape) 


class GGUFMoEMethod(FusedMoEMethodBase):
    """MoE method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        tensor_shape = (num_experts, 2 * intermediate_size_per_partition,
                        hidden_size)
        #gate up proj
        w13_qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            w13_qweight, {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
            })
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        layer.register_parameter("w13_qweight", w13_qweight)

        w13_qweight_type = Parameter(torch.empty(1, dtype=torch.uint8),
                                     requires_grad=False)
        set_weight_attrs(w13_qweight_type, {
            "is_gguf_weight_type": True,
            "weight_type": 0,
            "ignore_warning": True
        })
        set_weight_attrs(w13_qweight_type, extra_weight_attrs)
        layer.register_parameter("w13_qweight_type", w13_qweight_type)

        tensor_shape = (num_experts, intermediate_size_per_partition,
                        hidden_size)
        #gate down proj
        w2_qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            w2_qweight, {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
            })
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        layer.register_parameter("w2_qweight", w2_qweight)

        w2_qweight_type = Parameter(torch.empty(1, dtype=torch.uint8),
                                    requires_grad=False)
        set_weight_attrs(w2_qweight_type, {
            "is_gguf_weight_type": True,
            "weight_type": 0,
            "ignore_warning": True
        })

        set_weight_attrs(w2_qweight_type, extra_weight_attrs)
        layer.register_parameter("w2_qweight_type", w2_qweight_type)
        self.act = SiluAndMul()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ):
        assert activation == "silu", "Only SiLU activation is supported."
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "Apply router weight on input is not supported for"
                "fused GGUF MoE method.")

        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias)
        return _fused_moe_gguf(x, layer.w13_qweight, layer.w2_qweight,
                               topk_weights, topk_ids,
                               layer.w13_qweight_type.weight_type,
                               layer.w2_qweight_type.weight_type, self.act)


class GGUFEmbeddingMethod(GGUFLinearMethod):
    """Embedding method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def embedding(self, layer: torch.nn.Module,
                  x: torch.Tensor) -> torch.Tensor:
        weight = layer.weight
        return F.embedding(x, weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        qweight = layer.qweight
        qweight_type = layer.qweight_type
        
        params_dtype = getattr(qweight, "params_dtype", torch.get_default_dtype())
        weight = dequant_gguf(qweight.data.cpu().numpy().reshape(-1), qweight_type.weight_type, qweight.tensor_shape)
        
        del layer.qweight
        del layer.qweight_type
        layer.weight = torch.nn.Parameter(weight.to(params_dtype).to(qweight.device),
                                           requires_grad=False)
        assert layer.weight.is_cuda
        assert layer.weight.dtype == params_dtype
        if layer.__class__.__name__ == "ParallelLMHead":
            from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
            layer.quant_method = UnquantizedEmbeddingMethod()


class GGUFCPUInfer:
    cpuinfer = None
    
    def __init__(self) -> None:
        import os
        thread_num = int(os.environ.get("VLLM_GGUF_THREAD_NUMS", 33))
        if GGUFCPUInfer.cpuinfer is None:
            GGUFCPUInfer.cpuinfer = cpuinfer_ext.CPUInfer(thread_num)


class GGUFMoEMethod(FusedMoEMethodBase):
    CPUINfer = None
    CUR_STREAM = None
    INPUT_TENSOR_CPU = None
    TOPK_IDS_CPU = None
    TOPK_WEIGHT_CPU = None
    OUTPUT_CPU = None
    OUTPUT_GPU = None
    MAX_B = 0
    
    def __init__(self, quant_config: GGUFConfig, prefix: str):
        if GGUFMoEMethod.CPUINfer is None:
            if cpuinfer_ext is None:
                raise RuntimeError("ktransformers not installed!!")
            else:
                GGUFMoEMethod.CPUINfer = GGUFCPUInfer()
        self.quant_config = quant_config
        self.prefix = prefix

    def create_weights(self, layer: torch.nn.Module, 
                       num_experts: int,
                       hidden_size: int, 
                       intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, 
                       **extra_weight_attrs):
        
        gate_up_tensor_shape = (intermediate_size_per_partition, hidden_size, num_experts)
        down_tensor_shape = (hidden_size, intermediate_size_per_partition, num_experts)
        gate_weight = GGUFUninitializedParameter(requires_grad=False)
        up_weight = GGUFUninitializedParameter(requires_grad=False)
        down_weight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            gate_weight, {
                "tensor_shape": gate_up_tensor_shape,
                "is_gguf_weight": True,
                "data_container": []
            })
        set_weight_attrs(
            up_weight, {
                "tensor_shape": gate_up_tensor_shape,
                "is_gguf_weight": True,
                "data_container": []
            })
        set_weight_attrs(
            down_weight, {
                "tensor_shape": down_tensor_shape,
                "is_gguf_weight": True,
                "data_container": []
            })
        set_weight_attrs(gate_weight, extra_weight_attrs)
        set_weight_attrs(up_weight, extra_weight_attrs)
        set_weight_attrs(down_weight, extra_weight_attrs)
        layer.register_parameter("gate_weight", gate_weight)
        layer.register_parameter("up_weight", up_weight)
        layer.register_parameter("down_weight", down_weight)
        setattr(layer, "gate_type", None)
        setattr(layer, "up_type", None)
        setattr(layer, "down_type", None)


    def process_weights_after_loading(self, layer: torch.nn.Module, 
                                      all_tensors: gguf.ReaderTensor,
                                      gguf_to_hf_name_map: Dict[str, str]) -> None:
        all_done = [None] * 3
        ptrs = [None] * 3
        types = [None] * 3
        prefix = self.prefix.replace("model.layers","blk").replace(".mlp.experts","")
        for tensor in all_tensors:
            # if tensor.name in gguf_to_hf_name_map:
            if tensor.name == prefix + ".ffn_gate_exps.weight":
                data = tensor.data
                index = 0
                if all(all_done):
                    assert False, "repeat weights!!!"
            elif tensor.name == prefix + ".ffn_up_exps.weight":
                data = tensor.data
                index = 1
                if all(all_done):
                    assert False, "repeat weights!!!"
            elif tensor.name == prefix + ".ffn_down_exps.weight": 
                data = tensor.data
                index = 2
                if all(all_done):
                    assert False, "repeat weights!!!"
            else:
                index = -1
            if index != -1:
                types[index] = tensor.tensor_type
                ptrs[index] = ctypes.addressof(
                    ctypes.cast(data.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
                )
                all_done[index] = True      
        assert all(all_done)
        
        moe_config = MOEConfig(
            layer.num_experts,
            layer.top_k,
            layer.hidden_size,
            layer.intermediate_size_per_partition,
            64,
            10,
            1024,
            ptrs[0],
            ptrs[1],
            ptrs[2],
            types[0],
            types[1],
            types[2],
            30 if torch.get_default_dtype() == torch.bfloat16 else 1,
        )
        cpu_moe = MOE(moe_config)
        self.CPUINfer.cpuinfer.submit(cpu_moe.warm_up())
        self.CPUINfer.cpuinfer.sync()
        setattr(layer, "cpu_moe", cpu_moe)
        
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        extra_residual: Optional[torch.Tensor] = None,
        routed_scaling_factor: float = 1.0,
    ) -> torch.Tensor:
        B = x.shape[0]
        topk_ids = torch.empty([B, top_k], dtype=torch.long, device=x.device)
        
        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            topk_ids=topk_ids)        
        
        if GGUFMoEMethod.CUR_STREAM is None:
            GGUFMoEMethod.CUR_STREAM = torch.cuda.default_stream()
        if GGUFMoEMethod.MAX_B < B:
            GGUFMoEMethod.MAX_B = B
            GGUFMoEMethod.INPUT_TENSOR_CPU = torch.empty_like(x, dtype=x.dtype, device="cpu", pin_memory=True)
            GGUFMoEMethod.TOPK_IDS_CPU = torch.empty_like(topk_ids, dtype=torch.long, device="cpu",pin_memory=True)
            GGUFMoEMethod.TOPK_WEIGHT_CPU = torch.empty_like(topk_weights, dtype=topk_weights.dtype, device="cpu",pin_memory=True)
            GGUFMoEMethod.OUTPUT_CPU = torch.zeros_like(x, dtype=x.dtype, device="cpu",pin_memory=True)
            GGUFMoEMethod.OUTPUT_GPU = torch.zeros_like(x, dtype=x.dtype, device=x.device)
        
        input_tensor_cpu = GGUFMoEMethod.INPUT_TENSOR_CPU[:B]
        topk_ids_cpu = GGUFMoEMethod.TOPK_IDS_CPU[:B]
        topk_weights_cpu = GGUFMoEMethod.TOPK_WEIGHT_CPU[:B]
        output_cpu = GGUFMoEMethod.OUTPUT_CPU[:B]
        output_gpu = GGUFMoEMethod.OUTPUT_GPU[:B]

        input_tensor_cpu.copy_(x, non_blocking=True)
        topk_ids_cpu.copy_(topk_ids, non_blocking=True)
        topk_weights_cpu.copy_(topk_weights, non_blocking=True)
        GGUFMoEMethod.CUR_STREAM.synchronize()
        cpu_moe = layer.cpu_moe
        self.CPUINfer.cpuinfer.submit(
            cpu_moe.forward(topk_ids.size(0), topk_ids.size(1), topk_ids_cpu.data_ptr(), topk_weights_cpu.data_ptr(), input_tensor_cpu.data_ptr(), output_cpu.data_ptr()))
        self.CPUINfer.cpuinfer.sync()
        output_gpu.copy_(output_cpu, non_blocking=True)
        torch.cuda.default_stream().synchronize()
        return GGUFMoEMethod.OUTPUT_GPU[:B]


class GGUFUninitializedParameter(UninitializedParameter):
    cls_to_become = Parameter
    data_container: List[torch.Tensor]

    def materialize_nested(self) -> Parameter:
        dtype = {data.dtype for data in self.data_container}
        assert len(dtype) == 1, ValueError(
            f"Data container has mixed dtypes: {dtype}")
        dtype = next(iter(dtype))
        nested_data = torch.nested.nested_tensor(self.data_container,
                                                 device=self.device,
                                                 dtype=dtype)
        self.data_container.clear()
        param = torch.Tensor._make_subclass(self.cls_to_become,
                                            nested_data,
                                            require_grad=False)
        for k, v in self.__dict__.items():
            setattr(param, k, v)
        return param
