from ixformer.train.speedformer.models.chatglm.modeling_chatglm import RotaryEmbedding
from ixformer.train.speedformer.layers.rotary_pos_embedding import RotaryEmbedding


class ChatglmRotaryEmbedding(RotaryEmbedding):
    def from_native_attr(attr_class, *args, **kwargs):
        dim = attr_class.dim
        rote = RotaryEmbedding(dim=dim)
        return rote
