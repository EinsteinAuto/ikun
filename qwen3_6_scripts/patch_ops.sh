echo "[build] trigger 20260901"
echo "[build] trigger 202609011237"
echo "[build] trigger 202609011636"
echo "[build] trigger 202609011655"
echo "[build] trigger 202609011710"
echo "[build] trigger 202609031035"
echo "[build] trigger 202609031049"
#!/usr/bin/env bash
# BI-V100 patch script for Qwen3.6-35B-A3B (Qwen3_5 MoE architecture)
#
# Triton situation on BI-V100:
#   - Standard Triton 2.3.1 is already present in the image.
#   - HAS_TRITON = False (hardcoded in vendor vllm), but Triton is still used
#     for TP-mode cache management (custom_cache_manager / libentry).
#   - The vendor's triton_utils/__init__.py, custom_cache_manager.py, libentry.py
#     are already correct for standard Triton 2.3.1 — do NOT overwrite them.
#   - DO NOT install BI-V150 corex Triton 2.1.0 (pkgs/triton): that causes
#     GPU hang on BI-V100 because the Triton CUDA PTX kernels are incompatible.

# Recommended server start command for TP=4 support 256K, needs chunked prefill
# CUDA_VISIBLE_DEVICES="4,5,6,7" VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 python3 -m vllm.entrypoints.openai.api_server \
#     --model /workspace/models/Qwen3.6-35B-A3B --port 1111 --served-model-name llm \
#     --max-model-len 262144 --trust-remote-code -tp 4 --gpu-memory-utilization 0.90 \
#     --max-num-seqs 1 --disable-log-requests --disable-frontend-multiprocessing \
#     --max-num-batched-tokens 8192 --enable-chunked-prefill --enable-prefix-caching \
#     --max-seq-len-to-capture 32768 --enable-auto-tool-choice \
#     --tool-call-parser qwen3_coder --reasoning-parser qwen3
#
# With prefix caching (GDN align-mode, requires chunked prefill):
# CUDA_VISIBLE_DEVICES="4,5,6,7" VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 python3 -m vllm.entrypoints.openai.api_server \
#     --model /workspace/models/Qwen3.6-35B-A3B --port 1111 --served-model-name llm \
#     --max-model-len 262144 --trust-remote-code -tp 4 --gpu-memory-utilization 0.90 \
#     --max-num-seqs 1 --disable-log-requests --disable-frontend-multiprocessing \
#     --max-num-batched-tokens 8192 --enable-chunked-prefill --enable-prefix-caching \
#     --max-seq-len-to-capture 32768 --enable-auto-tool-choice \
#     --tool-call-parser qwen3_coder --reasoning-parser qwen3

set -eo pipefail

# cd into this script's directory so ./relative paths work
cd "$(dirname "${BASH_SOURCE[0]}")"
echo "[patch_ops] working directory: $(pwd)"

build_stage() { printf '[BI100 BUILD] %s\n' "$1" >&2; }
require_file() {
    local path=$1
    [[ -f "$path" ]] || {
        printf 'required patch source is missing: %s\n' "$path" >&2
        exit 2
    }
}
install_patch_file() {
    local source=$1
    local target=$2

    require_file "$source"
    mkdir -p "$(dirname "$target")"
    install -m 0644 "$source" "$target"
}

build_stage "patch script entered"

build_stage "patching torch._inductor triton_heuristics (get_cuda_stream compat)"
# --- torch._inductor: fix Triton 2.3.1 / corex torch incompatibility --------
# The corex torch build's triton_heuristics.py line 43 does:
#   from triton.runtime.jit import get_cuda_stream, KernelInterface
# But Standard Triton 2.3.1 (BI-V100 image) does not export get_cuda_stream.
# Upstream PyTorch >=2.5 uses torch._C._cuda_getCurrentRawStream instead.
# We patch the file on disk and inject a runtime compat shim.
python3 ./patch_triton_compat.py <<'PY_DISK_PATCH'
import sys
sys.path.insert(0, '.')
from patch_triton_compat import patch_triton_heuristics_on_disk
patched = patch_triton_heuristics_on_disk()
if not patched:
    print('[skip] triton_heuristics already patched or not found')
PY_DISK_PATCH

build_stage "bridging ixformer SDK 0.6.0 infer API to CoreX 3.2.3 _functions"
# SDK 0.6.0 inference/functions/*.py calls _C.infer.xxx but CoreX 3.2.3
# _C.so only exposes _C._functions.xxx_forward. Deploy the bridge module
# into VLLM_ROOT so env_override.py can find it at runtime.
if [[ -f "./patch_ixformer_infer.py" ]]; then
    cp -f "./patch_ixformer_infer.py" "${VLLM_ROOT}/patch_ixformer_infer.py"
    python3 -m py_compile "${VLLM_ROOT}/patch_ixformer_infer.py"
    echo "[ok] deployed ixformer infer bridge to ${VLLM_ROOT}/"
    # Pre-apply: run bridge now so any subsequent py_compile imports succeed
    python3 -c "import sys; sys.path.insert(0,'.'); import patch_ixformer_infer" 2>/dev/null || true
fi

build_stage "checking offline transformers dependency"
# --- transformers: Qwen3_5 tokenizer / model files --------------------------
TRANSFORMERS_REQUIRED_VERSION="4.55.3"
if ! python3 - "$TRANSFORMERS_REQUIRED_VERSION" <<'PY'
import importlib.metadata
import sys

required = sys.argv[1]
try:
    installed = importlib.metadata.version("transformers")
except importlib.metadata.PackageNotFoundError:
    raise SystemExit(1)
raise SystemExit(0 if installed == required else 1)
PY
then
  WHEEL_DIR="./wheels"
  if ! ls "${WHEEL_DIR}/transformers-${TRANSFORMERS_REQUIRED_VERSION}"*.whl >/dev/null 2>&1; then
    echo "transformers ${TRANSFORMERS_REQUIRED_VERSION} is required, but no offline wheel was found in ${WHEEL_DIR}" >&2
    exit 2
  fi
  python3 -m pip install --no-index --no-deps --find-links="${WHEEL_DIR}" \
    "transformers==${TRANSFORMERS_REQUIRED_VERSION}"
fi

python3 - "$TRANSFORMERS_REQUIRED_VERSION" <<'PY'
import importlib.metadata
import sys

required = sys.argv[1]
installed = importlib.metadata.version("transformers")
if installed != required:
    raise SystemExit(
        f"transformers version mismatch: expected {required}, got {installed}")
print(f"[ok] transformers {installed}")
PY

build_stage "discovering Python package roots"
python3 - <<'PY' > /tmp/qwen36_patch_paths.env
from patch_utils import package_root, shell_env_line

print(shell_env_line("VLLM_ROOT", package_root("vllm")))
print(shell_env_line("TRANSFORMERS_ROOT", package_root("transformers")))
PY
source /tmp/qwen36_patch_paths.env

echo "VLLM_ROOT=${VLLM_ROOT}"
echo "TRANSFORMERS_ROOT=${TRANSFORMERS_ROOT}"
[[ -d "$VLLM_ROOT" ]] || {
    printf 'vLLM root does not exist: %s\n' "$VLLM_ROOT" >&2
    exit 2
}

VLLM_OVERRIDE_ROOT="../vllm_overrides"
[[ -d "$VLLM_OVERRIDE_ROOT" ]] || {
    printf 'vLLM override directory missing: %s\n' "$VLLM_OVERRIDE_ROOT" >&2
    exit 2
}

build_stage "installing authoritative vLLM core block overrides"
# core/interfaces.py — removed: not present in vllm_overrides/
#   (vllm/core/interfaces.py is already the correct version)
# core/evictor_v2.py — removed: identical to vllm/core/evictor_v2.py
# core/block/cpu_kv_content_cache.py — removed: identical to vllm/ copy
install_patch_file \
    "${VLLM_OVERRIDE_ROOT}/core/block/cpu_gpu_block_allocator.py" \
    "${VLLM_ROOT}/core/block/cpu_gpu_block_allocator.py"
install_patch_file \
    "${VLLM_OVERRIDE_ROOT}/core/block/prefix_caching_block.py" \
    "${VLLM_ROOT}/core/block/prefix_caching_block.py"
install_patch_file \
    "${VLLM_OVERRIDE_ROOT}/core/block/block_table.py" \
    "${VLLM_ROOT}/core/block/block_table.py"
install_patch_file \
    "${VLLM_OVERRIDE_ROOT}/core/block_manager_v2.py" \
    "${VLLM_ROOT}/core/block_manager_v2.py"
install_patch_file \
    "${VLLM_OVERRIDE_ROOT}/sampling_params.py" \
    "${VLLM_ROOT}/sampling_params.py"
install_patch_file \
    "${VLLM_OVERRIDE_ROOT}/model_executor/sampling_metadata.py" \
    "${VLLM_ROOT}/model_executor/sampling_metadata.py"
install_patch_file \
    "${VLLM_OVERRIDE_ROOT}/model_executor/layers/sampler.py" \
    "${VLLM_ROOT}/model_executor/layers/sampler.py"

# BI100-DP data parallel overrides — removed: config.py, engine/arg_utils.py,
# engine/llm_engine.py, executor/mp_distributed_executor.py, worker/worker.py
# are not present in vllm_overrides/.  vllm/ already has the correct versions.

build_stage "installing hash-pinned CoreX 3.2.3 extensions"
bash ./install_prebuilt_corex.sh "${VLLM_ROOT}"

build_stage "installing BI100 runtime modules"
cp ./bi100_env.py "${VLLM_ROOT}/bi100_env.py"
cp ./bi100_profile.py "${VLLM_ROOT}/bi100_profile.py"
cp ./block_major_kv_cache.py "${VLLM_ROOT}/block_major_kv_cache.py"
cp ./gdn_prefix.py "${VLLM_ROOT}/gdn_prefix.py"
cp ./ep_fused_moe_patch.py "${VLLM_ROOT}/ep_fused_moe_patch.py"

build_stage "installing CoreX paged-KV swap compatibility"
cp ./_custom_ops.py "${VLLM_ROOT}/_custom_ops.py"
cp ./cache_engine.py "${VLLM_ROOT}/worker/cache_engine.py"
# worker swap order + block_major capacity + startup profile guard
# are pre-merged into vendor_overrides/vllm/worker/worker.py

# --- paged_attn.py: replace forward_prefix with pure-PyTorch fallback -------
# The Triton context_attention_fwd kernel hangs BI-V100 GPUs permanently
# (standard Triton 2.3.1 PTX is not supported by the corex runtime either).
# Our paged_attn.py bypasses it entirely via _forward_prefix_pytorch, which
# utilizes K-tiling techniques, and also have _forward_decode_pytorch to bypass kernel
# when context length is high
cp ./paged_attn.py "${VLLM_ROOT}/attention/ops/paged_attn.py"

# --- model_runner.py: fix prefix_cache_hit stays True in chunked-prefill chunk 2+ ---
# Bug: _compute_for_prefix_cache_hit Case 1 (prefix_cache_len <= context_len)
# leaves prefix_cache_hit=True. Then _add_seq_group uses block_table=computed_block_nums
# (only the original prefix blocks), ignoring chunk-1 KV cache blocks.
# _forward_prefix_pytorch then gets an undersized block_tables and crashes with
# "amax(): Expected reduction dim -1 to have non-zero size" on the 2nd tile.
# Fix: set prefix_cache_hit=False for Case 1 so the full block_tables is used.
cp ./model_runner.py "${VLLM_ROOT}/worker/model_runner.py"

build_stage "installing distributed module overrides (task 05/20)"
DIST_OVERRIDE_ROOT="./distributed_override"
if [[ -d "$DIST_OVERRIDE_ROOT" ]]; then
    # Top-level distributed files
    for f in __init__.py communication_op.py parallel_state.py utils.py; do
        install_patch_file "${DIST_OVERRIDE_ROOT}/${f}" "${VLLM_ROOT}/distributed/${f}"
    done

    # device_communicators (modified + new)
    for f in base_device_communicator.py cpu_communicator.py cuda_communicator.py \
             cuda_wrapper.py custom_all_reduce.py custom_all_reduce_utils.py \
             hpu_communicator.py neuron_communicator.py pynccl.py \
             pynccl_wrapper.py shm_broadcast.py tpu_communicator.py \
             xpu_communicator.py; do
        install_patch_file "${DIST_OVERRIDE_ROOT}/device_communicators/${f}" \
            "${VLLM_ROOT}/distributed/device_communicators/${f}"
    done

    # kv_transfer directory
    cp -r "${DIST_OVERRIDE_ROOT}/kv_transfer" "${VLLM_ROOT}/distributed/"

    # platforms (required by new distributed: get_device_communicator_cls, is_fully_connected)
    if [[ -d "${DIST_OVERRIDE_ROOT}/platforms" ]]; then
        for f in __init__.py interface.py cuda.py cpu.py rocm.py tpu.py xpu.py hpu.py neuron.py; do
            [[ -f "${DIST_OVERRIDE_ROOT}/platforms/${f}" ]] && \
                install_patch_file "${DIST_OVERRIDE_ROOT}/platforms/${f}" "${VLLM_ROOT}/platforms/${f}"
        done
    fi
fi

build_stage "installing executor startup diagnostics"
# executor startup debug + worker startup profile guard + block_major capacity
# are pre-merged into vendor_overrides and whole-file copies
cp ./multiproc_worker_utils.py "${VLLM_ROOT}/executor/multiproc_worker_utils.py"

build_stage "installing transformers Qwen3.5 model support"
cp -r ./qwen3_5 "${TRANSFORMERS_ROOT}/models/"
cp -r ./qwen3_5_moe "${TRANSFORMERS_ROOT}/models/"
python3 ./patch_transformers_qwen3_5.py

build_stage "installing vLLM Qwen3.6 model implementation"
# --- vllm model: Qwen3.6-35B-A3B (Qwen3_5 MoE arch) -------------------------
cp ./mamba_cache.py "${VLLM_ROOT}/model_executor/models/"
cp ./qwen3_5.py "${VLLM_ROOT}/model_executor/models/qwen3_5.py"
cp ./registry.py "${VLLM_ROOT}/model_executor/models/registry.py"
cp ./interfaces.py "${VLLM_ROOT}/model_executor/models/interfaces.py"
cp ./interfaces_base.py "${VLLM_ROOT}/model_executor/models/interfaces_base.py"

# --- sequence.py: fix completion_tokens inflation under chunked prefill ------
# Bug: get_output_token_ids_to_return(delta=True) with num_new_tokens=0
# returns _cached_all_token_ids[-0:] == [0:] (the ENTIRE prompt+output list).
# Each prefill chunk step adds prompt_len to previous_num_tokens, so a 10K
# prompt processed in 3 chunks inflates completion_tokens by ~30K.
# Also adds num_cached_tokens field to RequestMetrics for prefix-cache stats.
cp ./sequence.py "${VLLM_ROOT}/sequence.py"

# --- scheduler.py: record num_cached_tokens in RequestMetrics ----------------
# Reports only the longest prefix backed by both live KV blocks and an exact
# GDN restore state. Raw KV-only hits must not inflate cached_tokens.
# serving_chat.py exposes the value in the OpenAI-compatible usage details.
cp ./scheduler.py "${VLLM_ROOT}/core/scheduler.py"

build_stage "installing diagnostic initial allocation trace"
# block_manager_cache_trace is pre-merged into vendor_overrides/vllm/core/block_manager_v2.py
cp ./outputs.py "${VLLM_ROOT}/outputs.py"

build_stage "installing scheduler and attention patches"
# --- xformers: bypass cudnnFlashAttnForward (head_dim=256 > 128 limit) ------
# Injects _run_sdpa_fallback (pure matmul+softmax) into xformers.py.
# Required because head_dim=256 > 128 and ixformer flash attention either
# crashes (is_causal=True) or produces wrong output (attn_mask path).
# The fallback uses query_start_loc to derive actual query lengths, so it
# works correctly during profiling runs with chunked-prefill-style batches.
# also bypasses auto chunked prefill on
cp ./xformers.py "${VLLM_ROOT}/attention/backends/xformers.py"
cp ./logits_processor.py "${VLLM_ROOT}/model_executor/layers/logits_processor.py"
cp ./outlines_decoding.py "${VLLM_ROOT}/model_executor/guided_decoding/outlines_decoding.py"
# arg_utils.py xformers patches are pre-merged into vendor_overrides/vllm/engine/arg_utils.py
# bi100_timer profile instrumentation is pre-merged into xformers.py

build_stage "installing API parsers and serving modules"
# --- tool parser: Qwen3 XML tool call format ---------------------------------
# Registers "qwen3_coder" parser for Qwen3.6 XML-style tool calls:
#   <tool_call><function=name><parameter=key>\nvalue\n</parameter></function></tool_call>
# Use at server start: --tool-call-parser qwen3_coder --enable-auto-tool-choice
cp ./qwen3coder_tool_parser.py "${VLLM_ROOT}/entrypoints/openai/tool_parsers/"
cp ./tool_parsers__init__.py "${VLLM_ROOT}/entrypoints/openai/tool_parsers/__init__.py"

# --- reasoning parser: Qwen3 <think>...</think> split ------------------------
# Adds --reasoning-parser qwen3 support.
# Routes thinking tokens to reasoning_content, rest to content in the delta.
# Works together with --tool-call-parser qwen3_coder (think → tool call flow).
#
# PRD #69: Clear __pycache__ BEFORE copying patched .py files.
# Base image's compiled .pyc (protocol.py with extra="forbid") would otherwise
# shadow our patched .py, causing ~25% of requests to reject
# max_completion_tokens / reasoning_effort with HTTP 400.
find "${VLLM_ROOT}/entrypoints" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

cp -r ./reasoning "${VLLM_ROOT}/"
cp ./protocol.py "${VLLM_ROOT}/entrypoints/openai/protocol.py"
cp ./cli_args.py "${VLLM_ROOT}/entrypoints/openai/cli_args.py"
cp ./serving_chat.py "${VLLM_ROOT}/entrypoints/openai/serving_chat.py"
cp ./serving_tokenization.py \
    "${VLLM_ROOT}/entrypoints/openai/serving_tokenization.py"
cp ./api_server.py "${VLLM_ROOT}/entrypoints/openai/api_server.py"
cp ./chat_utils.py "${VLLM_ROOT}/entrypoints/chat_utils.py"
python3 - ./api_server.py \
        "${VLLM_ROOT}/entrypoints/openai/api_server.py" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_bytes()
installed = Path(sys.argv[2]).read_bytes()
if source != installed:
    raise SystemExit("runtime api_server overlay identity mismatch")
PY

# quantization, parameter.py, utils.py, fused_moe override blocks — removed:
# none of these subtrees/files exist in vllm_overrides/.
# vllm/model_executor/layers/quantization/, parameter.py, utils.py, and
# fused_moe/ are already the correct versions in the main vllm/ tree.

# transformers_utils, spec_decode, lora, prompt_adapter override blocks — removed:
# none of these subtrees exist in vllm_overrides/.
# vllm/transformers_utils/, spec_decode/, lora/, prompt_adapter/ are already
# the correct versions in the main vllm/ tree.

# PRD #69: Clear ALL __pycache__ under VLLM_ROOT after every cp/patch is done.
# py_compile below only compiles ./qwen3_6_scripts, not VLLM_ROOT, so this
# ensures the docker snapshot has no stale .pyc for any patched vllm module.
find "${VLLM_ROOT}" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# --- catch-all: deploy remaining vendor_overrides files -----------------------
# Root-level files (utils.py, jsontree.py, envs.py, etc.) and sub-modules
# (multimodal, compilation, inputs, guided_decoding, model_loader) that were
# prepared in vendor_overrides but not deployed by earlier explicit stages.
build_stage "deploying remaining vendor_overrides files"
(cd "${VLLM_OVERRIDE_ROOT}" && find . -name '*.py' -type f) | while read -r rel; do
    rel="${rel#./}"
    dst="${VLLM_ROOT}/${rel}"
    mkdir -p "$(dirname "$dst")"
    cp -f "${VLLM_OVERRIDE_ROOT}/${rel}" "$dst"
done

build_stage "compiling submission Python sources"
find . -path './wheels' -prune -o -name '*.py' -print0 | xargs -0 python3 -m py_compile
build_stage "patch script completed"
build_stage "installing ix_fused_moe 7-step pipeline and ex_engine"
cp ./ix_fused_moe.py "${VLLM_ROOT}/model_executor/models/ix_fused_moe.py"
if [[ -d "./ex_engine" ]]; then
    cp -rf ./ex_engine "${VLLM_ROOT}/../ex_engine"
    cp -rf ./ex_engine /workspace/ex_engine 2>/dev/null || true
fi
