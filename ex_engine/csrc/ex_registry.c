// ex_engine/csrc/ex_registry.c — EX Engine runtime: dlopen registry + dispatch
//
// CCCL parallel: cub/device/dispatch/dispatch_reduce.cuh Dispatch() selects
// policy by compute_capability then launches kernel. We select factor by
// hardware_id then call kernel_fn through the loaded .so.

#include "ex_engine.h"

#include <dlfcn.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Registry lifecycle
// ---------------------------------------------------------------------------

int ex_registry_init(ex_registry_t* reg, const ex_hardware_t* hw) {
    if (!reg || !hw) return -1;
    memset(reg, 0, sizeof(*reg));
    reg->hardware = *hw;
    return 0;
}

int ex_registry_load(ex_registry_t* reg, ex_factor_id_t id, const char* so_path) {
    if (!reg || !so_path || id < 0 || id >= EX_FACTOR_COUNT) return -1;

    // Close existing if reloading
    if (reg->handles[id]) {
        dlclose(reg->handles[id]);
        reg->handles[id] = NULL;
        reg->factors[id] = NULL;
    }

    void* handle = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        fprintf(stderr, "[EX] dlopen(%s) failed: %s\n", so_path, dlerror());
        return -1;
    }

    // Every .so must export "ex_get_factor"
    ex_get_factor_fn_t get_factor =
        (ex_get_factor_fn_t)dlsym(handle, "ex_get_factor");
    if (!get_factor) {
        fprintf(stderr, "[EX] dlsym(ex_get_factor) failed in %s: %s\n",
                so_path, dlerror());
        dlclose(handle);
        return -1;
    }

    ex_factor_t* factor = get_factor(&reg->hardware);
    if (!factor) {
        fprintf(stderr, "[EX] ex_get_factor returned NULL from %s\n", so_path);
        dlclose(handle);
        return -1;
    }

    // Verify factor_id matches what we requested
    if (factor->factor_id != id) {
        fprintf(stderr, "[EX] Factor ID mismatch: requested %d, got %d from %s\n",
                (int)id, (int)factor->factor_id, so_path);
        dlclose(handle);
        return -1;
    }

    reg->handles[id] = handle;
    reg->factors[id] = factor;
    reg->loaded_count++;

    fprintf(stderr, "[EX] Loaded factor %d (%s v%s) from %s | "
            "threads=%d items=%d vec=%d smem=%d\n",
            (int)id, factor->name, factor->version, so_path,
            factor->tuning.threads_per_block,
            factor->tuning.items_per_thread,
            factor->tuning.vec_size,
            factor->tuning.shared_mem_bytes);
    return 0;
}

// Factor .so naming convention: ex_factor_<id>.so
// e.g. ex_factor_0.so = MOE_TOPK_SOFTMAX
//      ex_factor_5.so = GDN_CHUNK_FWD
int ex_registry_load_dir(ex_registry_t* reg, const char* dir_path) {
    if (!reg || !dir_path) return -1;

    DIR* dir = opendir(dir_path);
    if (!dir) {
        fprintf(stderr, "[EX] Cannot open directory: %s\n", dir_path);
        return -1;
    }

    int loaded = 0;
    struct dirent* ent;
    while ((ent = readdir(dir)) != NULL) {
        // Match ex_factor_<N>.so
        int factor_id = -1;
        if (sscanf(ent->d_name, "ex_factor_%d.so", &factor_id) == 1 &&
            factor_id >= 0 && factor_id < EX_FACTOR_COUNT) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/%s", dir_path, ent->d_name);
            if (ex_registry_load(reg, (ex_factor_id_t)factor_id, path) == 0) {
                loaded++;
            }
        }
    }
    closedir(dir);

    fprintf(stderr, "[EX] Loaded %d/%d factors from %s\n",
            loaded, (int)EX_FACTOR_COUNT, dir_path);
    return loaded;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

int ex_dispatch(const ex_registry_t* reg, ex_factor_id_t id,
                void* output, const void* input,
                const void* aux_inputs[], int n_aux,
                const int64_t dims[], int n_dims,
                void* stream) {
    if (!reg || id < 0 || id >= EX_FACTOR_COUNT) return -1;

    const ex_factor_t* factor = reg->factors[id];
    if (!factor || !factor->kernel) return -1;

    return factor->kernel(output, input, aux_inputs, n_aux, dims, n_dims, stream);
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

void ex_registry_destroy(ex_registry_t* reg) {
    if (!reg) return;
    for (int i = 0; i < EX_FACTOR_COUNT; i++) {
        if (reg->handles[i]) {
            dlclose(reg->handles[i]);
            reg->handles[i] = NULL;
        }
        reg->factors[i] = NULL;
    }
    reg->loaded_count = 0;
}
