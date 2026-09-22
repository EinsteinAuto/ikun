// ex_engine/csrc/factor_gdn_chunk_fwd.cu
//
// Factor 5: GDN_CHUNK_FWD — GatedDeltaNet chunked prefill forward
//
// CCCL reference: cub/device/dispatch/tuning/tuning_scan.cuh
//   ScanLookbackPolicy with decoupled lookback for streaming prefix ops.
//   GDN is fundamentally a recurrent scan: state[t] = decay * state[t-1] + write
//
// The NaN problem (from dockerrizhi.txt):
//   "NaN in prefill GatedDeltaNet layer 0 (frac=0.9998), replacing with zeros"
//   Root cause: _torch_chunk_gated_delta_rule does cumsum on gate values
//   that can overflow float16 range. The FlashQLA SM70 kernel compiled but
//   also produced NaN because it uses float16 accumulators.
//
// Fix: Full float32 accumulation in the recurrent state update.
//   state = beta * (k ⊗ v) + exp(gate) * state   [all in fp32]
//   output = (q @ state).to(fp16)                  [cast only at output]
//
// BI-V100 tuning (SM70, 16 SMs):
//   chunk_size = 16 (reduced from 64 to prevent overflow)
//   head_dim = 128
//   num_heads = 2 per TP rank (8 total / 4 TP)
//   SMEM: state matrix = 128×128×4 = 64KB → won't fit in 48KB SMEM
//   Solution: Tile state update, keep running state in registers/global

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

extern "C" {
#include "ex_engine.h"
}

// ---------------------------------------------------------------------------
// GDN Recurrent state update kernel (one CTA per head)
//
// For each chunk of tokens:
//   For each time step t in chunk:
//     decay = exp(gate[t])          — scalar per head
//     beta_t = sigmoid(beta[t])     — scalar per head
//     k_t = key[t]                  — (D,) vector
//     v_t = value[t]                — (D,) vector
//     state = decay * state + beta_t * outer(k_t, v_t)   — (D, D) matrix
//     output[t] = query[t] @ state  — (D,) vector
//
// State matrix is D×D = 128×128 = 16K floats = 64KB in fp32.
// Cannot fit in SMEM (48KB). Use register tiling: each thread owns
// a (D/TILE) × (D/TILE) block of the state matrix.
// ---------------------------------------------------------------------------

static constexpr int HEAD_DIM = 128;
static constexpr int CHUNK_SIZE = 16;

// Tile config: 256 threads, each owns a 8×8 block of state
// 128/8 = 16 tiles per dim → 16×16 = 256 tiles = 256 threads ✓
static constexpr int TILE = 8;
static constexpr int TILES_PER_DIM = HEAD_DIM / TILE;  // 16
static constexpr int BLOCK_THREADS = TILES_PER_DIM * TILES_PER_DIM;  // 256

__global__ void gdn_chunk_fwd_kernel(
    half* __restrict__    output,    // (B, L, H, D)
    float* __restrict__   state_out, // (B, H, D, D) — updated state
    const half* __restrict__  query, // (B, L, H, D)
    const half* __restrict__  key,   // (B, L, H, D)
    const half* __restrict__  value, // (B, L, H, D)
    const float* __restrict__ gate,  // (B, L, H)
    const float* __restrict__ beta,  // (B, L, H)
    const float* __restrict__ state_in, // (B, H, D, D) — initial state
    int B, int L, int H, int D
) {
    // Block: (batch, head) pair
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    if (b >= B) return;

    int tid = threadIdx.x;
    int tile_row = tid / TILES_PER_DIM;  // which row tile (0..15)
    int tile_col = tid % TILES_PER_DIM;  // which col tile (0..15)

    // Each thread owns TILE×TILE = 8×8 = 64 floats of state
    float my_state[TILE][TILE];

    // Load initial state
    int row_start = tile_row * TILE;
    int col_start = tile_col * TILE;
    const float* sin = state_in + (b * H + h) * D * D;
    #pragma unroll
    for (int r = 0; r < TILE; r++) {
        #pragma unroll
        for (int c = 0; c < TILE; c++) {
            my_state[r][c] = sin[(row_start + r) * D + (col_start + c)];
        }
    }

    // Shared memory for broadcast: one time step at a time
    __shared__ float s_k[HEAD_DIM];    // current key vector
    __shared__ float s_v[HEAD_DIM];    // current value vector
    __shared__ float s_decay;          // exp(gate)
    __shared__ float s_beta;           // sigmoid(beta)

    // Process each time step sequentially (recurrent)
    for (int t = 0; t < L; t++) {
        // Thread 0 loads gate, beta; all threads load their k/v slice
        if (tid == 0) {
            float g = gate[(b * L + t) * H + h];
            float bt = beta[(b * L + t) * H + h];
            // Clamp gate to prevent overflow: exp(88) ≈ FLT_MAX for float32
            g = fminf(fmaxf(g, -20.0f), 20.0f);
            s_decay = expf(g);
            s_beta = 1.0f / (1.0f + expf(-bt));  // sigmoid
        }

        // Cooperatively load k and v vectors into SMEM
        if (tid < D) {
            int idx = ((b * L + t) * H + h) * D + tid;
            s_k[tid] = __half2float(key[idx]);
            s_v[tid] = __half2float(value[idx]);
        }
        __syncthreads();

        float decay = s_decay;
        float bt = s_beta;

        // State update: state = decay * state + beta * outer(k, v)
        // Each thread updates its TILE×TILE block
        #pragma unroll
        for (int r = 0; r < TILE; r++) {
            float k_r = s_k[row_start + r];
            #pragma unroll
            for (int c = 0; c < TILE; c++) {
                float v_c = s_v[col_start + c];
                my_state[r][c] = decay * my_state[r][c] + bt * k_r * v_c;
            }
        }

        // Query @ state → output[t]
        // Each thread computes partial dot product for its tile rows
        // output[d] = sum_j query[j] * state[d][j]
        // Thread (tile_row, tile_col) has state[row_start..+TILE][col_start..+TILE]
        // It contributes: for each r in 0..TILE-1:
        //   partial[row_start+r] += sum_{c=0..TILE-1} query[col_start+c] * state[r][c]

        // Load query
        __shared__ float s_q[HEAD_DIM];
        if (tid < D) {
            int idx = ((b * L + t) * H + h) * D + tid;
            s_q[tid] = __half2float(query[idx]);
        }
        __syncthreads();

        // Compute partial result for my tile rows
        float partial[TILE];
        #pragma unroll
        for (int r = 0; r < TILE; r++) {
            partial[r] = 0.0f;
            #pragma unroll
            for (int c = 0; c < TILE; c++) {
                partial[r] += s_q[col_start + c] * my_state[r][c];
            }
        }

        // Reduce across col tiles (threads with same tile_row, different tile_col)
        // Use shared memory: each thread writes its partial, then tile_col=0 sums
        __shared__ float s_partials[TILES_PER_DIM][TILES_PER_DIM][TILE];
        // s_partials[tile_row][tile_col][r]
        #pragma unroll
        for (int r = 0; r < TILE; r++) {
            s_partials[tile_row][tile_col][r] = partial[r];
        }
        __syncthreads();

        // tile_col == 0 aggregates across all col tiles
        if (tile_col == 0) {
            float result[TILE];
            #pragma unroll
            for (int r = 0; r < TILE; r++) {
                result[r] = 0.0f;
                #pragma unroll
                for (int tc = 0; tc < TILES_PER_DIM; tc++) {
                    result[r] += s_partials[tile_row][tc][r];
                }
            }
            // Write output
            int out_base = ((b * L + t) * H + h) * D + row_start;
            #pragma unroll
            for (int r = 0; r < TILE; r++) {
                output[out_base + r] = __float2half(result[r]);
            }
        }
        __syncthreads();
    }

    // Write final state
    float* sout = state_out + (b * H + h) * D * D;
    #pragma unroll
    for (int r = 0; r < TILE; r++) {
        #pragma unroll
        for (int c = 0; c < TILE; c++) {
            sout[(row_start + r) * D + (col_start + c)] = my_state[r][c];
        }
    }
}

// ---------------------------------------------------------------------------
// Factor dispatch
// ---------------------------------------------------------------------------

static int gdn_chunk_fwd_dispatch(
    void*          output,
    const void*    input,
    const void*    aux_inputs[],
    int            n_aux,
    const int64_t  dims[],
    int            n_dims,
    void*          stream
) {
    // dims = {B, L, H, D}
    // input = query (B, L, H, D) half
    // aux[0] = key, aux[1] = value, aux[2] = gate (float), aux[3] = beta (float)
    // aux[4] = state_in (B, H, D, D) float
    // aux[5] = state_out (B, H, D, D) float (output)
    if (n_dims < 4 || n_aux < 6) return -1;

    int B = (int)dims[0];
    int L = (int)dims[1];
    int H = (int)dims[2];
    int D = (int)dims[3];

    if (D != HEAD_DIM) return -1;  // Only support D=128

    half*  out      = (half*)output;
    const half*  q  = (const half*)input;
    const half*  k  = (const half*)aux_inputs[0];
    const half*  v  = (const half*)aux_inputs[1];
    const float* g  = (const float*)aux_inputs[2];
    const float* bt = (const float*)aux_inputs[3];
    const float* si = (const float*)aux_inputs[4];
    float* so       = (float*)aux_inputs[5];

    cudaStream_t cu_stream = (cudaStream_t)stream;

    // Dynamic SMEM: s_partials needs TILES_PER_DIM × TILES_PER_DIM × TILE × sizeof(float)
    //             = 16 × 16 × 8 × 4 = 8192 bytes
    //             + s_k, s_v, s_q = 3 × 128 × 4 = 1536 bytes
    //             + s_decay, s_beta = 8 bytes
    //             Total ≈ 9736 bytes << 48KB ✓

    dim3 grid(B * H);
    dim3 block(BLOCK_THREADS);  // 256

    gdn_chunk_fwd_kernel<<<grid, block, 0, cu_stream>>>(
        out, so, q, k, v, g, bt, si, B, L, H, D
    );

    return 0;
}

// ---------------------------------------------------------------------------
// .so export
// ---------------------------------------------------------------------------

static ex_factor_t s_factor;

extern "C" ex_factor_t* ex_get_factor(const ex_hardware_t* hw) {
    s_factor.factor_id   = EX_FACTOR_GDN_CHUNK_FWD;
    s_factor.name        = "gdn_chunk_fwd";
    s_factor.version     = "1.0.0";
    s_factor.tuning      = (ex_tuning_t){
        .threads_per_block = BLOCK_THREADS,  // 256
        .items_per_thread  = TILE * TILE,    // 64 (state elements per thread)
        .vec_size          = 1,
        .shared_mem_bytes  = 10240,  // ~10KB
        .num_warps         = 8,
        .num_stages        = 1  // sequential recurrence, no pipelining
    };
    s_factor.kernel          = gdn_chunk_fwd_dispatch;
    s_factor.kernel_fallback = NULL;
    return &s_factor;
}
