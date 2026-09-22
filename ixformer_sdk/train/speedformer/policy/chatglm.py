from typing import Callable, Dict, List, Union
from torch.nn import Module

from ixformer.train.speedformer.policy.utils import SubModuleReplacementDescription
from ixformer.train.speedformer.policy.replacer import Replacer

from ixformer.train.speedformer.layers.normalization import APEXFusedRMSNorm, IXFFusedRMSNorm
from ixformer.train.speedformer.layers.chatglm.attention import ChatglmFlashAttention
from ixformer.train.speedformer.layers.chatglm.methods import ChatGLMModel_forward


class ChatglmReplacer(Replacer):
    def __init__(self):
        self.policy = {}

    def module_policy(self) -> Dict[str | Module, List[SubModuleReplacementDescription]]:
        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="final_layernorm",
                    target_module=APEXFusedRMSNorm,
                    kwargs={}
                ),
            ],
            target_key="GLMTransformer"
        )
        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="input_layernorm",
                    target_module=APEXFusedRMSNorm,
                    kwargs={}
                ),
                SubModuleReplacementDescription(
                    suffix="post_attention_layernorm",
                    target_module=APEXFusedRMSNorm,
                    kwargs={}
                ),
            ],
            target_key="GLMBlock"
        )
        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="self_attention",
                    target_module=ChatglmFlashAttention,
                    kwargs={}
                ),
            ],
            target_key="GLMBlock"
        )
        self.append_or_create_method_replacement(
            description=[
                {"forward": ChatGLMModel_forward()}
            ],
            target_key="ChatGLMModel"
        )
