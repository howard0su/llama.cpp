#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// if C99 - static_assert is noop
// ref: https://stackoverflow.com/a/53923785/4039976
#ifndef static_assert
#define static_assert(cond, msg) struct global_scope_noop_trick
#endif

//
// logging
//

#if (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG(...)
#endif

#if (GGML_DEBUG >= 5)
#define GGML_PRINT_DEBUG_5(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_5(...)
#endif

#if (GGML_DEBUG >= 10)
#define GGML_PRINT_DEBUG_10(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_10(...)
#endif

#define GGML_PRINT(...) printf(__VA_ARGS__)

#define GGML_ASSERT(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "GGML_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

// available tensor operations:
enum ggml_op {
    GGML_OP_NONE = 0,

    GGML_OP_DUP,
    GGML_OP_ADD,
    GGML_OP_SUB,
    GGML_OP_MUL,
    GGML_OP_DIV,
    GGML_OP_SQR,
    GGML_OP_SQRT,
    GGML_OP_SUM,
    GGML_OP_MEAN,
    GGML_OP_REPEAT,
    GGML_OP_ABS,
    GGML_OP_SGN,
    GGML_OP_NEG,
    GGML_OP_STEP,
    GGML_OP_RELU,
    GGML_OP_GELU,
    GGML_OP_SILU,
    GGML_OP_NORM, // normalize
    GGML_OP_RMS_NORM,

    GGML_OP_MUL_MAT,

    GGML_OP_SCALE,
    GGML_OP_CPY,
    GGML_OP_CONT,
    GGML_OP_RESHAPE,
    GGML_OP_VIEW,
    GGML_OP_PERMUTE,
    GGML_OP_TRANSPOSE,
    GGML_OP_GET_ROWS,
    GGML_OP_DIAG_MASK_INF,
    GGML_OP_SOFT_MAX,
    GGML_OP_ROPE,
    GGML_OP_CONV_1D_1S,
    GGML_OP_CONV_1D_2S,

    GGML_OP_FLASH_ATTN,
    GGML_OP_FLASH_FF,

    GGML_OP_MAP_UNARY,
    GGML_OP_MAP_BINARY,

    GGML_OP_COUNT,
};


// ggml object
struct ggml_object {
    size_t offs;
    size_t size;

    struct ggml_object * next;

    char padding[8];
};

static const size_t GGML_OBJECT_SIZE = sizeof(struct ggml_object);

// n-dimensional tensor
struct ggml_tensor {
    enum ggml_type type;

    int    n_dims;
    int64_t ne[GGML_MAX_DIMS]; // number of elements
    size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
                               // nb[0] = sizeof(type)
                               // nb[1] = nb[0]   * ne[0] + padding
                               // nb[i] = nb[i-1] * ne[i-1]

    // compute data
    enum ggml_op op;

    bool is_param;

    struct ggml_tensor * grad;
    struct ggml_tensor * src0;
    struct ggml_tensor * src1;
    struct ggml_tensor * opt[GGML_MAX_OPT];

    // thread scheduling
    int n_tasks;

    // performance
    int     perf_runs;
    int64_t perf_cycles;
    int64_t perf_time_us;

    void * data;
    struct ggml_context * ctx;
};

// computation graph
struct ggml_cgraph {
    int n_nodes;
    int n_leafs;
    int n_threads;

    size_t work_size;
    struct ggml_tensor * work;

    struct ggml_tensor * nodes[GGML_MAX_NODES];
    struct ggml_tensor * grads[GGML_MAX_NODES];
    struct ggml_tensor * leafs[GGML_MAX_NODES];

    // performance
    int     perf_runs;
    int64_t perf_cycles;
    int64_t perf_time_us;
};

/////////////////////////////////////////////////////////////////////////////////

static inline int ggml_nrows(const struct ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}


///////////////////////////////////////////////////////////////////////////////
// CUDA related functions
///////////////////////////////////////////////////////////////////////////////
extern int cuda_device; // -1: No CUDA, 0-N: one cuda card

int ggml_cuda_init(int prefered);
void *ggml_cuda_allocate(int size);
void ggml_cuda_copy(void * target, const void * src, size_t size);


struct ggml_tensor * ggml_set_f32_cuda(struct ggml_tensor * tensor, float value);
struct ggml_tensor * ggml_set_i32_cuda(struct ggml_tensor * tensor, int32_t value);

void ggml_compute_forward_cuda(struct ggml_compute_params * params, struct ggml_tensor * tensor);

#ifdef __cplusplus
}
#endif