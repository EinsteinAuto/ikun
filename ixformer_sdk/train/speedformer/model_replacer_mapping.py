import torch

from ixformer.train.speedformer.policy.gpt2 import GPT2Replacer
from ixformer.train.speedformer.policy.qwen2 import Qwen2Replacer
from ixformer.train.speedformer.policy.llama import LlamaReplacer
from ixformer.train.speedformer.policy.baichuan import BaichuanReplacer
from ixformer.train.speedformer.policy.bloom import BloomReplacer
from ixformer.train.speedformer.policy.chatglm import ChatglmReplacer


ModelMapping = {
    "gpt2": GPT2Replacer,
    "qwen2": Qwen2Replacer,
    "llama": LlamaReplacer,
    "baichuan": BaichuanReplacer,
    "bloom": BloomReplacer,
    "chatglm": ChatglmReplacer
}
