#pragma once

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
    char padding[8];
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
