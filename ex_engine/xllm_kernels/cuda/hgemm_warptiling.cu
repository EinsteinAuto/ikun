// hgemm_warptiling.cu — FP16 warp-tiling GEMM for BI-V100 (warp_size=64)
//
// 1:1 from siboehm/SGEMM_CUDA kernel 10 (sgemmWarptiling).
// Changes from original:
//   1. WARPSIZE = 32 → 64  (BI-V100 confirmed)
//   2. float → __half for A/B/C data and shared memory
//   3. float4 vectorized load → 4 scalar __half loads
//   4. threadResults accumulator stays float (FP32 accumulation)
//   5. C writeback: scalar instead of float4

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const int WARPSIZE = 64; // BI-V100 confirmed

namespace wt {
template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const __half *A, const __half *B,
                             __half *As, __half *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB) {
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    // Load 4 halfs from A, transpose while storing
    __half a0 = A[(innerRowA + offset) * K + innerColA * 4 + 0];
    __half a1 = A[(innerRowA + offset) * K + innerColA * 4 + 1];
    __half a2 = A[(innerRowA + offset) * K + innerColA * 4 + 2];
    __half a3 = A[(innerRowA + offset) * K + innerColA * 4 + 3];
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = a0;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = a1;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = a2;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = a3;
  }

  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    // Load 4 halfs from B, no transpose
    Bs[(innerRowB + offset) * BN + innerColB * 4 + 0] =
        B[(innerRowB + offset) * N + innerColB * 4 + 0];
    Bs[(innerRowB + offset) * BN + innerColB * 4 + 1] =
        B[(innerRowB + offset) * N + innerColB * 4 + 1];
    Bs[(innerRowB + offset) * BN + innerColB * 4 + 2] =
        B[(innerRowB + offset) * N + innerColB * 4 + 2];
    Bs[(innerRowB + offset) * BN + innerColB * 4 + 3] =
        B[(innerRowB + offset) * N + innerColB * 4 + 3];
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults, const __half *As,
                const __half *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // populate registers for whole warptile
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] = __half2float(
            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i]);
      }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] = __half2float(
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i]);
      }
    }

    // execute warptile matmul — FP32 accumulation
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                          (wSubColIdx * TN) + resIdxN] +=
                regM[wSubRowIdx * TM + resIdxM] *
                regN[wSubColIdx * TN + resIdxN];
          }
        }
      }
    }
  }
}

} // namespace wt

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    hgemmWarptiling(int M, int N, int K, float alpha, const __half *A,
                    const __half *B, float beta, __half *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // Placement of the warp in the threadblock tile
  const uint warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  // size of the warp subtile
  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER;
  constexpr uint WSUBN = WN / WNITER;

  // Placement of the thread in the warp subtile
  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 63]
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

  // allocate space for the current blocktile in SMEM
  __shared__ __half As[BM * BK];
  __shared__ __half Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  // Move C_ptr to warp's output tile
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // calculating the indices that this thread will load into SMEM
  // FP16: 4 halfs per thread per step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
  // we cache into registers on the warptile level
  float regM[WMITER * TM] = {0.0f};
  float regN[WNITER * TN] = {0.0f};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    __syncthreads();
    wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                        TN>(regM, regN, threadResults, As, Bs, warpRow, warpCol,
                            threadRowInWarp, threadColInWarp);
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
    __syncthreads();
  }

  // write out the results — scalar writeback (no float4 for __half)
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      __half *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 1) {
          uint idx = (threadRowInWarp * TM + resIdxM) * N +
                     threadColInWarp * TN + resIdxN;
          float c_old = __half2float(C_interim[idx]);
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          C_interim[idx] = __float2half(alpha * threadResults[i] + beta * c_old);
        }
      }
    }
  }
}


// ============================================================================
// Launch wrapper
// ============================================================================
void launch_hgemm_warptiling(
    int M, int N, int K,
    float alpha,
    const __half* A,
    const __half* B,
    float beta,
    __half* C,
    cudaStream_t stream)
{
    // Config B — best on BI-V100 (beats cublas 0.7x on 256x4096@4096x11008):
    // probe_k10_configs.sh confirmed: 7.6ms vs cublas 10.5ms
    // 128 threads = 2 warps of 64
    // WMITER = (64*64)/(64*8*4*2) = 4096/4096 = 1
    // WSUBM = 64/1 = 64, WSUBN = 64/2 = 32
    // threads_per_warp = (64/8)*(32/4) = 8*8 = 64 ✓
    constexpr int NUM_THREADS = 128;
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int WM = 64, WN = 64;
    constexpr int WNITER = 2;
    constexpr int TM = 8, TN = 4;

    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 block(NUM_THREADS);

    hgemmWarptiling<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
}
