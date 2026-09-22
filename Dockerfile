FROM harbor.4pd.io/modelhubxc/enginex-iluvatar/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3
RUN mkdir -p /workspace
WORKDIR /workspace/
COPY ./qwen3_6_scripts /workspace/qwen3_6_scripts
COPY ./computility-run.yaml /workspace/computility-run.yaml
COPY ./vllm /workspace/vllm
COPY ./ixformer_sdk/inference/functions /workspace/ixformer_functions
COPY ./ixformer_inference_functions_init.py /workspace/ixformer_inference_functions_init.py
RUN chmod +x /workspace/qwen3_6_scripts/patch_ops.sh && \
    bash /workspace/qwen3_6_scripts/patch_ops.sh 2>&1 | tee /workspace/patch_ops.log ; \
    echo "[Dockerfile] patch_ops exit code: $?"
RUN VLLM_ROOT=/usr/local/corex/lib64/python3/dist-packages/vllm && \
    IXFORMER_ROOT=/usr/local/corex/lib64/python3/dist-packages/ixformer && \
    cp -rf /workspace/vllm/* "${VLLM_ROOT}/" && \
    find "${VLLM_ROOT}" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    cp -r /workspace/ixformer_functions "${IXFORMER_ROOT}/inference/functions" && \
    cp /workspace/ixformer_inference_functions_init.py "${IXFORMER_ROOT}/inference/functions/__init__.py" && \
    mkdir -p "${IXFORMER_ROOT}/contrib/vllm_flash_attn" && \
    echo 'from ixformer import flash_attn_varlen_func, flash_attn_func, flash_attn_padded_func' > "${IXFORMER_ROOT}/contrib/vllm_flash_attn/__init__.py" && \
    pip3 install blake3 --break-system-packages 2>/dev/null && \
    echo "[overlay] done"
