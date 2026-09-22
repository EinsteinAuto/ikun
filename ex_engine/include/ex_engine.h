// ex_engine/include/ex_engine.h — EX Engine: Algorithm Factor Replacement via dlopen
//
// Architecture mirrors CCCL's dispatch pattern:
//   CCCL:  compute_capability → policy_selector → {threads, items, vec_size} → kernel
//   EX:    hardware_id        → factor_table    → {op_fn_ptr, tuning_params} → dlopen .so
//
// The base image (BI-V100 corex SDK) has ixformer with gaps:
//   PRESENT in ixformer.functions:
//     silu_and_mul, gelu_and_mul, rms_norm, fused_add_rms_norm,
//     vllm_rotary_embedding_neox, vllm_single_query_cached_kv_attention (v1/v2),
//     vllm_cache_ops_reshape_and_cache, vllm_swap_blocks, vllm_copy_cache
//
//   MISSING from ixformer.functions (every call falls back to slow PyTorch):
//     vllm_moe_topk_softmax        — MoE routing, called 36× per token per layer
//     vllm_moe_align_block_size    — MoE block alignment
//     vllm_invoke_fused_moe_kernel — MoE expert GEMM fusion
//     gelu_tanh_and_mul            — activation variant
//     batched_rotary_embedding     — batch RoPE
//
// This engine provides .so replacements for each missing factor, compiled for
// BI-V100's SM70-class architecture using the corex clang/16 toolchain.

#ifndef EX_ENGINE_H
#define EX_ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

// ============================================================================
// Hardware descriptor (CCCL compute_capability equivalent)
// ============================================================================
typedef struct {
    int sm_major;           // SM version major (BI-V100 = 7)
    int sm_minor;           // SM version minor (BI-V100 = 0)
    int sm_count;           // Number of SMs (BI-V100 = 16)
    int max_threads_per_sm; // Max resident threads per SM
    int shared_mem_per_sm;  // Shared memory per SM in bytes (49152)
    int l2_cache_size;      // L2 cache size in bytes
    int memory_bus_width;   // Memory bus width in bits
    float memory_bandwidth; // GB/s (BI-V100 ≈ 56 GB/s per SM)
} ex_hardware_t;

// ============================================================================
// Tuning policy (CCCL ReducePassPolicy / ScanPolicy equivalent)
// ============================================================================
typedef struct {
    int threads_per_block;
    int items_per_thread;
    int vec_size;
    int shared_mem_bytes;   // SMEM budget (BI-V100 max 49152)
    int num_warps;
    int num_stages;         // Pipeline stages (1 = no async, 2 = SW pipeline)
} ex_tuning_t;

// ============================================================================
// Factor IDs — each represents one algorithm factor to replace
// Maps directly to the missing ixformer.functions ops
// ============================================================================
typedef enum {
    // MoE factors (P0 — called 36× per layer, 64 layers)
    EX_FACTOR_MOE_TOPK_SOFTMAX     = 0,  // topk + softmax routing
    EX_FACTOR_MOE_ALIGN_BLOCK      = 1,  // block alignment for scatter
    EX_FACTOR_MOE_FUSED_GEMM       = 2,  // fused expert GEMM

    // Activation factors (P1)
    EX_FACTOR_GELU_TANH_MUL        = 3,  // gelu_tanh_and_mul

    // RoPE factors (P1)
    EX_FACTOR_BATCHED_ROTARY       = 4,  // batched rotary embedding

    // GDN factors (P0 — 4 GDN layers produce NaN without proper kernel)
    EX_FACTOR_GDN_CHUNK_FWD        = 5,  // GatedDeltaNet chunked prefill
    EX_FACTOR_GDN_RECURRENT        = 6,  // GatedDeltaNet single-step decode

    // Cache factors (P2)
    EX_FACTOR_CACHE_APPEND          = 7,  // paged_attention_cache_appended
    EX_FACTOR_RESHAPE_CACHE_FLASH   = 8,  // reshape_and_cache_flash

    EX_FACTOR_COUNT                 = 9
} ex_factor_id_t;

// ============================================================================
// Factor entry point — each .so exports this struct
// ============================================================================

// Generic function pointer for the kernel dispatch
typedef int (*ex_kernel_fn_t)(
    void*          output,        // output tensor data_ptr
    const void*    input,         // primary input tensor data_ptr
    const void*    aux_inputs[],  // auxiliary inputs (weights, etc.)
    int            n_aux,         // number of auxiliary inputs
    const int64_t  dims[],        // tensor dimensions
    int            n_dims,        // number of dimensions
    void*          stream         // CUDA stream
);

// Each .so exports exactly one of these
typedef struct {
    ex_factor_id_t  factor_id;
    const char*     name;           // human-readable name
    const char*     version;        // semver string
    ex_tuning_t     tuning;         // tuned parameters for this hardware
    ex_kernel_fn_t  kernel;         // the replacement kernel
    ex_kernel_fn_t  kernel_fallback; // PyTorch reference (NULL = no fallback)
} ex_factor_t;

// Standard entry point name for dlopen: "ex_get_factor"
typedef ex_factor_t* (*ex_get_factor_fn_t)(const ex_hardware_t* hw);

// ============================================================================
// Factor registry — manages loaded .so factors
// ============================================================================
typedef struct {
    ex_factor_t*    factors[EX_FACTOR_COUNT];
    void*           handles[EX_FACTOR_COUNT];  // dlopen handles
    ex_hardware_t   hardware;
    int             loaded_count;
} ex_registry_t;

// Initialize registry with hardware info
int ex_registry_init(ex_registry_t* reg, const ex_hardware_t* hw);

// Load a single factor .so
int ex_registry_load(ex_registry_t* reg, ex_factor_id_t id, const char* so_path);

// Load all .so files from a directory
int ex_registry_load_dir(ex_registry_t* reg, const char* dir_path);

// Dispatch: call the loaded factor kernel, or return -1 if not loaded
int ex_dispatch(const ex_registry_t* reg, ex_factor_id_t id,
                void* output, const void* input,
                const void* aux_inputs[], int n_aux,
                const int64_t dims[], int n_dims,
                void* stream);

// Cleanup
void ex_registry_destroy(ex_registry_t* reg);

// ============================================================================
// BI-V100 default hardware descriptor
// ============================================================================
static inline ex_hardware_t ex_bi_v100_hardware(void) {
    return (ex_hardware_t){
        .sm_major           = 7,
        .sm_minor           = 0,
        .sm_count           = 16,
        .max_threads_per_sm = 2048,
        .shared_mem_per_sm  = 49152,
        .l2_cache_size      = 6 * 1024 * 1024,  // 6MB
        .memory_bus_width   = 4096,
        .memory_bandwidth   = 900.0f  // ~900 GB/s total
    };
}

#ifdef __cplusplus
}
#endif

#endif // EX_ENGINE_H
