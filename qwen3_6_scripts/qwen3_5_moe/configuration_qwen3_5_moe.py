# Adapted from transformers 5.2.0 for compatibility with transformers 4.55.3 + torch 2.1.0
# Source: transformers/models/qwen3_5_moe/configuration_qwen3_5_moe.py
# Stubs layer_type_validation and RopeParameters which do not exist in 4.55.3
# Removes ignore_keys_at_rope_validation / base_model_tp_plan / base_model_pp_plan
# which are 5.x-only and irrelevant for vLLM inference.

import os
from typing import Optional

from ...configuration_utils import PretrainedConfig as PreTrainedConfig

# --- Local stubs for APIs not present in transformers 4.55.3 ---
def layer_type_validation(layer_types, num_hidden_layers=None, attention=True):
    allowed = {"full_attention", "linear_attention"}
    if not all(lt in allowed for lt in layer_types):
        raise ValueError(f"layer_types entries must be in {allowed}, got {layer_types}")
    if num_hidden_layers is not None and num_hidden_layers != len(layer_types):
        raise ValueError(
            f"num_hidden_layers ({num_hidden_layers}) != len(layer_types) ({len(layer_types)})"
        )


HYBRID_KV_ACCOUNTING_ENV = "BI100_HYBRID_KV_ACCOUNTING"
HYBRID_KV_ACCOUNTING_CONFIG = "bi100_hybrid_kv_accounting_mode"
LEGACY_KV_ACCOUNTING = "legacy40"
FULL_ATTENTION_KV_ACCOUNTING = "full_attention"


def _hybrid_kv_accounting_mode(environ=None, serialized_mode=None):
    source = os.environ if environ is None else environ
    environment_mode = source.get(HYBRID_KV_ACCOUNTING_ENV)
    if (environment_mode is not None and serialized_mode is not None
            and environment_mode != serialized_mode):
        raise RuntimeError(
            f"{HYBRID_KV_ACCOUNTING_ENV}={environment_mode!r} conflicts "
            f"with serialized {HYBRID_KV_ACCOUNTING_CONFIG}="
            f"{serialized_mode!r}")
    mode = environment_mode or serialized_mode or LEGACY_KV_ACCOUNTING
    if mode not in (LEGACY_KV_ACCOUNTING, FULL_ATTENTION_KV_ACCOUNTING):
        raise RuntimeError(
            f"{HYBRID_KV_ACCOUNTING_ENV} must be "
            f"'{LEGACY_KV_ACCOUNTING}' or "
            f"'{FULL_ATTENTION_KV_ACCOUNTING}', got {mode!r}")
    return mode


def _vllm_layers_block_type(
    layer_types,
    environ=None,
    serialized_mode=None,
):
    """Expose hybrid-layer ownership in the form vLLM 0.6.3 consumes."""
    mode = _hybrid_kv_accounting_mode(environ, serialized_mode)
    if mode == LEGACY_KV_ACCOUNTING:
        return ["attention"] * len(layer_types)
    return [
        "attention" if layer_type == "full_attention" else layer_type
        for layer_type in layer_types
    ]

try:
    from typing import TypedDict
except ImportError:
    RopeParameters = dict
else:
    class RopeParameters(TypedDict, total=False):
        rope_theta: float
        rope_type: str
        partial_rotary_factor: float
        factor: float

# --- End stubs ---


class Qwen3_5MoeTextConfig(PreTrainedConfig):
    r"""
    Configuration for the text backbone of Qwen3.5-MoE / Qwen3.6-35B-A3B models.
    model_type is "qwen3_5_moe_text" (used internally by the nested config).
    """

    model_type = "qwen3_5_moe_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=248320,
        hidden_size=2048,
        num_hidden_layers=40,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=256,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts_per_tok=8,
        num_experts=256,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        layer_types=None,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        **kwargs,
    ):
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        self.rope_parameters = rope_parameters
        kwargs.setdefault("partial_rotary_factor", 0.25)

        self.layer_types = layer_types
        if self.layer_types is None:
            interval_pattern = kwargs.get("full_attention_interval", 4)
            self.layer_types = [
                "linear_attention" if bool((i + 1) % interval_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        super().__init__(**kwargs)


class Qwen3_5MoeVisionConfig(PreTrainedConfig):
    model_type = "qwen3_5_moe"

    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=3584,
        num_position_embeddings=2304,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range


class Qwen3_5MoeConfig(PreTrainedConfig):
    r"""
    Top-level configuration for Qwen3.5-MoE / Qwen3.6-35B-A3B.
    model_type = "qwen3_5_moe" matches the model card / config.json.
    Wraps Qwen3_5MoeTextConfig (and optionally Qwen3_5MoeVisionConfig).
    For vLLM text-only inference only text_config is consumed.
    """

    model_type = "qwen3_5_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        tie_word_embeddings=False,
        **kwargs,
    ):
        serialized_mode = kwargs.pop(HYBRID_KV_ACCOUNTING_CONFIG, None)
        serialized_layers = kwargs.pop("layers_block_type", None)
        if isinstance(text_config, dict):
            self.text_config = Qwen3_5MoeTextConfig(**text_config)
        elif text_config is None:
            self.text_config = Qwen3_5MoeTextConfig()
        else:
            self.text_config = text_config

        if isinstance(vision_config, dict):
            self.vision_config = Qwen3_5MoeVisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = Qwen3_5MoeVisionConfig()
        else:
            self.vision_config = vision_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)
        mode = _hybrid_kv_accounting_mode(
            serialized_mode=serialized_mode)
        layers_block_type = _vllm_layers_block_type(
            self.text_config.layer_types, serialized_mode=mode)
        if (serialized_layers is not None
                and list(serialized_layers) != layers_block_type):
            raise RuntimeError(
                "serialized layers_block_type conflicts with "
                f"{HYBRID_KV_ACCOUNTING_CONFIG}={mode!r}")
        setattr(self, HYBRID_KV_ACCOUNTING_CONFIG, mode)
        self.layers_block_type = layers_block_type


__all__ = ["Qwen3_5MoeConfig", "Qwen3_5MoeTextConfig"]
