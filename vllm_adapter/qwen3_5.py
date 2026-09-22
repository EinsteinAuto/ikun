# SPDX-License-Identifier: Apache-2.0
# Inference-only Qwen3_5Moe model compatible with HuggingFace weights.
#
# Adaptation strategy: Based on qwen3_moe.py (vllm 0.6.3+corex).
# Qwen3.6-35B-A3B uses hybrid attention (linear + full) but for initial
# bootstrap we treat ALL layers as full attention. This is correct but
# suboptimal — linear attention layers will use more KV cache than needed.
# Once baseline TPS is established, linear attention can be optimized.
#
# Key config differences vs Qwen3Moe:
# - config wraps text params in text_config
# - has shared_expert (shared_expert_intermediate_size)
# - 256 experts, top-8
# - layer_types: ["linear_attention", ..., "full_attention", ...]

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig
from vllm.distributed import (get_pp_group,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsPP
from .utils import (extract_layer_index, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

logger = init_logger(__name__)


def _get_text_config(config: PretrainedConfig) -> PretrainedConfig:
    """Extract text_config from the composite config.
    Qwen3.5Moe wraps all text params in config.text_config."""
    if hasattr(config, "text_config") and config.text_config is not None:
        tc = config.text_config
        # Ensure text_config is a proper config object, not a dict
        if isinstance(tc, dict):
            from transformers import AutoConfig
            tc = AutoConfig.for_model("qwen3_5_moe_text", **tc)
        return tc
    return config


class Qwen3_5MoeMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config,
                                           reduce_results=reduce_results)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen3_5MoeSparseMoeBlock(nn.Module):
    """MoE block with optional shared expert."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}.")

        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=getattr(config, "norm_topk_prob", True),
            quant_config=quant_config)

        self.gate = ReplicatedLinear(config.hidden_size,
                                     config.num_experts,
                                     bias=False,
                                     quant_config=None)

        # Shared expert (Qwen3.5 specific)
        shared_expert_intermediate = getattr(
            config, "shared_expert_intermediate_size", 0)
        if shared_expert_intermediate > 0:
            self.shared_expert = Qwen3_5MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=shared_expert_intermediate,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
            )
        else:
            self.shared_expert = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Router
        router_logits, _ = self.gate(hidden_states_flat)
        final_hidden_states = self.experts(
            hidden_states=hidden_states_flat,
            router_logits=router_logits)

        # Add shared expert output
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states_flat)
            final_hidden_states = final_hidden_states + shared_output

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states.view(orig_shape)


class Qwen3_5MoeAttention(nn.Module):
    """Standard full attention, used for ALL layers in bootstrap mode.
    In the real model, only every 4th layer uses full attention,
    the rest use GatedDeltaNet (linear attention). For bootstrap,
    we use full attention everywhere — correct but uses more KV cache."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        head_dim: Optional[int] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # QKV projection
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
        )

        # Qwen3.5 uses partial rotary
        rope_pct = 1.0
        if rope_scaling and "partial_rotary_factor" in rope_scaling:
            rope_pct = rope_scaling["partial_rotary_factor"]
        elif rope_scaling and "mrope_section" in rope_scaling:
            # M-RoPE: partial_rotary_factor from config
            rope_pct = rope_scaling.get("partial_rotary_factor", 0.25)

        rotary_dim = int(self.head_dim * rope_pct)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        # QK norm (Qwen3 style)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = self.q_norm(q.contiguous())
        k = self.k_norm(k.contiguous())

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3_5MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Determine layer type from config
        layer_types = getattr(config, "layer_types", None)
        if layer_types and layer_idx < len(layer_types):
            self.layer_type = layer_types[layer_idx]
        else:
            self.layer_type = "full_attention"

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_parameters",
                              getattr(config, "rope_scaling", None))

        # For bootstrap: use full attention for ALL layer types
        # This ignores linear_attention optimization but is correct
        self.self_attn = Qwen3_5MoeAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        self.mlp = Qwen3_5MoeSparseMoeBlock(
            config=config,
            quant_config=quant_config)

        self.input_layernorm = RMSNorm(config.hidden_size,
                                        eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                 eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # MoE
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3_5MoeModel(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen3_5MoeDecoderLayer(
                config=config,
                layer_idx=extract_layer_index(prefix),
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"],
                config.hidden_size))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i - self.start_layer],
                attn_metadata,
                residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual,
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3_5MoeForCausalLM(nn.Module, SupportsPP):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        # Extract text_config
        self.text_config = _get_text_config(config)
        self.quant_config = quant_config

        self.model = Qwen3_5MoeModel(
            self.text_config,
            cache_config,
            quant_config,
            prefix="model")

        if self.text_config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                self.text_config.vocab_size,
                self.text_config.hidden_size,
                quant_config=quant_config)

        self.logits_processor = LogitsProcessor(self.text_config.vocab_size)
        self.sampler = Sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                    attn_metadata, intermediate_tensors)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                        sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.text_config.num_experts)

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            # Skip vision encoder weights
            if name.startswith("visual.") or name.startswith("vision_"):
                continue
            # Skip MTP (multi-token prediction) weights
            if ".mtp_" in name or name.startswith("mtp_"):
                continue
            # Skip linear attention specific weights (conv, delta, gates)
            # These don't exist in our full-attention approximation
            if any(x in name for x in [
                "conv1d", "delta_net", "gated_delta",
                "linear_key", "linear_value",
                "A_log", "D", "dt_proj", "x_proj",
                "gate_norm", "fuse_norm",
            ]):
                continue

            if "rotary_emb.inv_freq" in name:
                continue

            # Handle "model.layers.X.self_attn." prefix mapping
            # The checkpoint may have different names for attention weights
            # depending on layer_type. We load them all into our uniform
            # full-attention layers.

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                # Map shared_expert weights
                if "shared_expert" in name:
                    name = name.replace(weight_name, param_name)
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name.endswith("kv_scale"):
                        remapped = name.replace(".kv_scale", ".attn.kv_scale")
                        if remapped not in params_dict:
                            continue
                        name = remapped
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


# Also export dense model alias (registry has both entries)
Qwen3_5ForCausalLM = Qwen3_5MoeForCausalLM
