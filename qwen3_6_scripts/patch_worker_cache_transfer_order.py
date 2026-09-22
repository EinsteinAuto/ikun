from patch_utils import package_root, replace_once


WORKER = package_root("vllm") / "worker" / "worker.py"

CLEAN_BLOCK = """\
        if (worker_input.blocks_to_swap_in is not None
                and worker_input.blocks_to_swap_in.numel() > 0):
            self.cache_engine[virtual_engine].swap_in(
                worker_input.blocks_to_swap_in)
        if (worker_input.blocks_to_swap_out is not None
                and worker_input.blocks_to_swap_out.numel() > 0):
            self.cache_engine[virtual_engine].swap_out(
                worker_input.blocks_to_swap_out)
"""

ORDERED_BLOCK = """\
        # BI100 content-addressed CPU KV tier may preserve a victim and reuse
        # that same GPU slot in one step. Complete every D2H before any H2D.
        if (worker_input.blocks_to_swap_out is not None
                and worker_input.blocks_to_swap_out.numel() > 0):
            self.cache_engine[virtual_engine].swap_out(
                worker_input.blocks_to_swap_out)
        if (worker_input.blocks_to_swap_in is not None
                and worker_input.blocks_to_swap_in.numel() > 0):
            self.cache_engine[virtual_engine].swap_in(
                worker_input.blocks_to_swap_in)
"""


replace_once(
    WORKER,
    CLEAN_BLOCK,
    ORDERED_BLOCK,
    required=True,
    already_contains="Complete every D2H before any H2D",
)
