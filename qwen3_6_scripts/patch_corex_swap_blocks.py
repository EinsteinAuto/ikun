from patch_utils import package_root, replace_once


CUSTOM_OPS = package_root("vllm") / "_custom_ops.py"

CLEAN_BLOCK = """\
def swap_blocks(src: torch.Tensor, dst: torch.Tensor,
                block_mapping: torch.Tensor) -> None:
    ixf_F.swap_blocks(src, dst, block_mapping)
"""

COMPATIBLE_BLOCK = """\
def swap_blocks(src: torch.Tensor, dst: torch.Tensor,
                block_mapping: torch.Tensor) -> None:
    # BI100 CoreX 3.2.3 exposes vllm_swap_blocks, while this vLLM build calls
    # the newer swap_blocks name. Normalize the worker's CPU int64 [N, 2]
    # tensor only for the legacy public API and fail fast on malformed maps.
    native_swap_blocks = getattr(ixf_F, "swap_blocks", None)
    if native_swap_blocks is not None:
        native_swap_blocks(src, dst, block_mapping)
        return

    vendor_swap_blocks = getattr(ixf_F, "vllm_swap_blocks", None)
    if vendor_swap_blocks is None:
        raise RuntimeError(
            "ixformer exposes neither swap_blocks nor vllm_swap_blocks")

    if isinstance(block_mapping, torch.Tensor):
        if block_mapping.device.type != "cpu":
            raise ValueError("swap block mapping must be a CPU tensor")
        if block_mapping.dtype != torch.int64:
            raise ValueError("swap block mapping must use torch.int64")
        if block_mapping.dim() != 2 or block_mapping.shape[1] != 2:
            raise ValueError("swap block mapping must have shape [N, 2]")
        pairs = block_mapping.tolist()
    elif isinstance(block_mapping, dict):
        pairs = list(block_mapping.items())
    else:
        raise TypeError("swap block mapping must be a tensor or dict")

    normalized_mapping = {}
    destinations = set()
    for source, destination in pairs:
        source = int(source)
        destination = int(destination)
        if source < 0 or destination < 0:
            raise ValueError("swap block indices must be non-negative")
        if source in normalized_mapping:
            raise ValueError(f"duplicate swap source block: {source}")
        if destination in destinations:
            raise ValueError(
                f"duplicate swap destination block: {destination}")
        normalized_mapping[source] = destination
        destinations.add(destination)
    vendor_swap_blocks(src, dst, normalized_mapping)
"""


replace_once(
    CUSTOM_OPS,
    CLEAN_BLOCK,
    COMPATIBLE_BLOCK,
    required=True,
    already_contains="BI100 CoreX 3.2.3 exposes vllm_swap_blocks",
)