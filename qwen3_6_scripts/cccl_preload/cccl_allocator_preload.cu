/*
 * cccl_allocator_preload.cu
 *
 * LD_PRELOAD .so — CUB CachingDeviceAllocator from CCCL upstream.
 * Full dependency chain (288 files) extracted into include/.
 *
 * Intercepts cudaMalloc/cudaFree, routes through CUB's geometric-bin
 * caching allocator. Strips expandable_segments from
 * PYTORCH_CUDA_ALLOC_CONF before libtorch reads it.
 *
 * Source: CCCL cub/cub/util_allocator.cuh (BSD-3, NVIDIA)
 * Build:  bash build_cccl_preload.sh
 */

/* ---- CCCL include chain (288 files from cccl_upstream) ---- */
#include <cub/util_allocator.cuh>

/* ---- System ---- */
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

/* ========================================================================
 * Configuration for BI-V100 (32GB × 4 cards)
 *
 * CUB CachingDeviceAllocator parameters:
 *   bin_growth = 2   (power-of-2 bins: 256B, 512B, 1KB, ... 4GB)
 *   min_bin    = 8   (2^8 = 256B minimum allocation)
 *   max_bin    = 32  (2^32 = 4GB maximum cached bin)
 *   max_cached = 8GB per device
 *
 * More granular bins (growth=2) than CUB default (growth=8) because
 * PyTorch tensor sizes vary widely in inference.
 * ======================================================================== */

static constexpr unsigned int ALLOC_BIN_GROWTH   = 2;
static constexpr unsigned int ALLOC_MIN_BIN      = 8;   /* 256 bytes */
static constexpr unsigned int ALLOC_MAX_BIN      = 32;  /* 4 GB */
static constexpr size_t       ALLOC_MAX_CACHED   = (size_t)8 * 1024 * 1024 * 1024; /* 8GB */

/* ---- Global allocator singleton ---- */
static cccl_preload::cub::CachingDeviceAllocator& get_allocator() {
    static cccl_preload::cub::CachingDeviceAllocator instance(
        ALLOC_BIN_GROWTH,
        ALLOC_MIN_BIN,
        ALLOC_MAX_BIN,
        ALLOC_MAX_CACHED,
        true  /* skip_cleanup: CoreX may tear down CUDA before our dtor */
    );
    return instance;
}

static bool g_preload_active = false;
static bool g_debug = false;

/* ---- Real cudaMalloc/cudaFree via dlsym(RTLD_NEXT) ---- */
using RealMalloc_t = cudaError_t (*)(void**, size_t);
using RealFree_t   = cudaError_t (*)(void*);

static RealMalloc_t get_real_malloc() {
    static RealMalloc_t fn = (RealMalloc_t)dlsym(RTLD_NEXT, "cudaMalloc");
    return fn;
}
static RealFree_t get_real_free() {
    static RealFree_t fn = (RealFree_t)dlsym(RTLD_NEXT, "cudaFree");
    return fn;
}

/* ========================================================================
 * Constructor: runs at LD_PRELOAD load time
 * ======================================================================== */
__attribute__((constructor))
static void cccl_preload_init() {
    const char* debug_env = getenv("CCCL_ALLOC_DEBUG");
    g_debug = (debug_env && atoi(debug_env) > 0);

    const char* disable_env = getenv("CCCL_ALLOC_DISABLE");
    if (disable_env && atoi(disable_env) > 0) {
        fprintf(stderr, "[cccl_alloc] DISABLED by CCCL_ALLOC_DISABLE=1\n");
        return;
    }

    /* Strip expandable_segments from PYTORCH_CUDA_ALLOC_CONF */
    const char* alloc_conf = getenv("PYTORCH_CUDA_ALLOC_CONF");
    if (alloc_conf) {
        std::string conf(alloc_conf);
        std::string clean;
        size_t pos = 0;
        while (pos < conf.size()) {
            size_t comma = conf.find(',', pos);
            if (comma == std::string::npos) comma = conf.size();
            std::string token = conf.substr(pos, comma - pos);
            if (token.find("expandable_segments") == std::string::npos) {
                if (!clean.empty()) clean += ",";
                clean += token;
            }
            pos = comma + 1;
        }
        if (clean.empty())
            unsetenv("PYTORCH_CUDA_ALLOC_CONF");
        else
            setenv("PYTORCH_CUDA_ALLOC_CONF", clean.c_str(), 1);

        fprintf(stderr, "[cccl_alloc] PYTORCH_CUDA_ALLOC_CONF: \"%s\" -> \"%s\"\n",
                alloc_conf, clean.empty() ? "(unset)" : clean.c_str());
    }

    /* Initialize allocator */
    auto& alloc = get_allocator();
    if (g_debug) {
        alloc.debug = true;
    }

    g_preload_active = true;
    fprintf(stderr,
        "[cccl_alloc] LD_PRELOAD active — CUB CachingDeviceAllocator "
        "(growth=%u, bins=[%u..%u], max_cached=%.1fGB)\n",
        ALLOC_BIN_GROWTH, ALLOC_MIN_BIN, ALLOC_MAX_BIN,
        (double)ALLOC_MAX_CACHED / (1024.0*1024.0*1024.0));
}

/* ========================================================================
 * cudaMalloc / cudaFree intercepts
 *
 * CUB's DeviceAllocate internally calls cudaMalloc on cache miss.
 * We must detect this reentrant call and forward to the real function,
 * otherwise we get infinite recursion → segfault.
 * ======================================================================== */

static thread_local bool g_in_allocator = false;

extern "C" cudaError_t cudaMalloc(void** devPtr, size_t size)
{
    if (!g_preload_active || g_in_allocator) {
        return get_real_malloc()(devPtr, size);
    }
    g_in_allocator = true;
    cudaError_t err = get_allocator().DeviceAllocate(devPtr, size);
    g_in_allocator = false;
    return err;
}

extern "C" cudaError_t cudaFree(void* devPtr)
{
    if (!g_preload_active || devPtr == nullptr || g_in_allocator) {
        return get_real_free()(devPtr);
    }
    g_in_allocator = true;
    cudaError_t err = get_allocator().DeviceFree(devPtr);
    g_in_allocator = false;
    return err;
}
