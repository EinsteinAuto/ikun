// hgemm_blocktiling.cu — FP16 GEMM for BI-V100
//
// 1:1 from siboehm/SGEMM_CUDA kernel 6 (sgemmVectorize).
// Changes: float→__half, float4→load 4 halfs, FP32 accumulator.
// No WARPSIZE usage. No cooperative_groups. CUDA 10.2 safe.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void hgemmVectorize(int M, int N, int K, float alpha,
                                const __half *A, const __half *B,
                                float beta, __half *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  // A stored transposed: As[BK][BM], B normal: Bs[BK][BN]
  __shared__ __half As[BM * BK];
  __shared__ __half Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  // FP16: load 4 halfs (8 bytes) per step.  4 halfs per thread.
  // siboehm: float4 = 4 floats = 128bit.  We do 4 halfs = 64bit.
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  // allocate thread-local cache for results in registerfile
  // FP32 accumulation to avoid FP16 precision loss
  float threadResults[TM * TN] = {0.0f};
  __half regM[TM];
  __half regN[TN];

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    // transpose A while loading it (same as siboehm)
    // Load 4 halfs from A
    __half a0 = A[innerRowA * K + innerColA * 4 + 0];
    __half a1 = A[innerRowA * K + innerColA * 4 + 1];
    __half a2 = A[innerRowA * K + innerColA * 4 + 2];
    __half a3 = A[innerRowA * K + innerColA * 4 + 3];
    As[(innerColA * 4 + 0) * BM + innerRowA] = a0;
    As[(innerColA * 4 + 1) * BM + innerRowA] = a1;
    As[(innerColA * 4 + 2) * BM + innerRowA] = a2;
    As[(innerColA * 4 + 3) * BM + innerRowA] = a3;

    // Load 4 halfs from B (no transpose)
    Bs[innerRowB * BN + innerColB * 4 + 0] = B[innerRowB * N + innerColB * 4 + 0];
    Bs[innerRowB * BN + innerColB * 4 + 1] = B[innerRowB * N + innerColB * 4 + 1];
    Bs[innerRowB * BN + innerColB * 4 + 2] = B[innerRowB * N + innerColB * 4 + 2];
    Bs[innerRowB * BN + innerColB * 4 + 3] = B[innerRowB * N + innerColB * 4 + 3];
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      // FP32 accumulation
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        float aVal = __half2float(regM[resIdxM]);
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              aVal * __half2float(regN[resIdxN]);
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 1) {
      uint row = cRow * BM + threadRow * TM + resIdxM;
      uint col = cCol * BN + threadCol * TN + resIdxN;
      if (row < M && col < N) {
        float c_old = __half2float(C[(threadRow * TM + resIdxM) * N +
                                     threadCol * TN + resIdxN]);
        C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
            __float2half(alpha * threadResults[resIdxM * TN + resIdxN] +
                         beta * c_old);
      }
    }
  }
}


// ============================================================================
// Launch wrapper — matches siboehm runSgemmVectorize
// ============================================================================
void launch_hgemm_blocktiling(
    int M, int N, int K,
    const __half* alpha_ptr,
    const __half* A, int lda,
    const __half* B, int ldb,
    const __half* beta_ptr,
    __half* C, int ldc,
    cudaStream_t stream)
{
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    // 256 threads — same as siboehm
    constexpr int NUM_THREADS = (BM * BN) / (TM * TN);

    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 block(NUM_THREADS);

    float alpha = 1.0f, beta = 0.0f;
    if (alpha_ptr) alpha = __half2float(*alpha_ptr);
    if (beta_ptr)  beta  = __half2float(*beta_ptr);

    hgemmVectorize<BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
}


// ============================================================================
// MoE expert GEMM — C++ loop over experts (replaces Python for-loop)
// ============================================================================
void launch_moe_expert_hgemm(
    int num_experts,
    const int* expert_counts,     // host, [num_experts]
    const int* expert_offsets,    // host, [num_experts]
    int N, int K,
    const __half* input,          // (total_tokens, K)
    const __half* weights,        // (num_experts, N, K)
    __half* output,               // (total_tokens, N)
    cudaStream_t stream)
{
    for (int e = 0; e < num_experts; e++) {
        int M_e = expert_counts[e];
        if (M_e == 0) continue;

        int off = expert_offsets[e];
        const __half* A = input + off * K;
        const __half* B = weights + (long long)e * N * K;
        __half* C_e = output + off * N;

        launch_hgemm_blocktiling(M_e, N, K,
                                 nullptr, A, K, B, N, nullptr, C_e, N, stream);
    }
}
