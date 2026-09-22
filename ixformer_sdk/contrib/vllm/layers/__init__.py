from .llama import forward_smoothquant
from .mixtral import mixtral_decoder_layer_forward

SUPPORT_REPLACE_METHOD = {
    "llama": forward_smoothquant,
}

SUPPORT_REPLACE_LAYER = {
    "llama": None,
}


def get_replace_forward(name: str):
    try:
        method = SUPPORT_REPLACE_METHOD[name]
    except:
        raise ValueError(
            f"Only support replace names: {SUPPORT_REPLACE_METHOD.keys()}, but got {name}"
        )
    return method


def get_replace_layer(name: str):
    try:
        layer = SUPPORT_REPLACE_LAYER[name]
    except:
        raise ValueError(
            f"Only support replace names: {SUPPORT_REPLACE_LAYER.keys()}, but got {name}"
        )
    return layer
