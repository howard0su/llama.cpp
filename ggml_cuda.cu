#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_fp16.h"

#include <assert.h>
#include <stdio.h>

#include "ggml.h"
#include "ggml-internal.h"

int cuda_device = -1; // -1: No CUDA, 0-N: one cuda card

int ggml_cuda_init(int prefered)
{
    int count;
    cudaGetDeviceCount(&count);

    if (prefered < count)
        cuda_device = prefered;
    else
        cuda_device = 0;

    cudaError_t cudaStatus = cudaSetDevice(cuda_device);
    if (cudaStatus != cudaSuccess) {
        return -1;
    }

    return 0;
}

void* ggml_cuda_allocate(int size)
{
    void *data;
    cudaError_t cudaStatus = cudaMalloc(&data, size);
    if (cudaStatus != cudaSuccess) {
        GGML_PRINT("%s: not enough VRAM (needed %zu)\n",
            __func__, size);
        assert(false);
        return NULL;
    }

    return data;
}

void ggml_cuda_copy(void * target, const void * src, size_t size)
{
    cudaError_t cudaStatus;
    cudaStatus = cudaMemcpy(target, src, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        GGML_PRINT("%s: Failed to do the copy", __func__);
        assert(false);
    }
}

/////////////////////////////////////////////////////////////////////////////

// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 512;

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}


template <typename Dtype>
__global__ void set_kernel(const int n, Dtype* y, const Dtype alpha) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

////////////////////////////////////////////////////////////////////////////////

struct ggml_tensor * ggml_set_f32_cuda(struct ggml_tensor * tensor, float value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = (char * const)tensor->data;

    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    // ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                    set_kernel<int8_t><<<CAFFE_GET_BLOCKS(nc), CAFFE_CUDA_NUM_THREADS>>>(
                        nc, (int8_t *)(data + i*n1), (int8_t)value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    set_kernel<int16_t><<<CAFFE_GET_BLOCKS(nc), CAFFE_CUDA_NUM_THREADS>>>(
                        nc, (int16_t *)(data + i*n1), (int16_t)value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    set_kernel<int32_t><<<CAFFE_GET_BLOCKS(nc), CAFFE_CUDA_NUM_THREADS>>>(
                        nc, (int32_t *)(data + i*n1), (int32_t)value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    set_kernel<ggml_fp16_t><<<CAFFE_GET_BLOCKS(nc), CAFFE_CUDA_NUM_THREADS>>>(
                        nc, (ggml_fp16_t *)(data + i*n1), (ggml_fp16_t)value);
                }
            } break;
        case GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    set_kernel<float><<<CAFFE_GET_BLOCKS(nc), CAFFE_CUDA_NUM_THREADS>>>(
                        nc, (float *)(data + i*n1), (float)value);
                }
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }

    return tensor;
}

struct ggml_tensor * ggml_set_i32_cuda(struct ggml_tensor * tensor, int32_t value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = (char * const)tensor->data;

    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    // ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                    set_kernel<int8_t><<<CAFFE_GET_BLOCKS(nc), CAFFE_CUDA_NUM_THREADS>>>(
                        nc, (int8_t *)(data + i*n1), (int8_t)value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    set_kernel<int16_t><<<CAFFE_GET_BLOCKS(nc), CAFFE_CUDA_NUM_THREADS>>>(
                        nc, (int16_t *)(data + i*n1), (int16_t)value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    set_kernel<int32_t><<<CAFFE_GET_BLOCKS(nc), CAFFE_CUDA_NUM_THREADS>>>(
                        nc, (int32_t *)(data + i*n1), (int32_t)value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    set_kernel<ggml_fp16_t><<<CAFFE_GET_BLOCKS(nc), CAFFE_CUDA_NUM_THREADS>>>(
                        nc, (ggml_fp16_t *)(data + i*n1), (ggml_fp16_t)value);
                }
            } break;
        case GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    set_kernel<float><<<CAFFE_GET_BLOCKS(nc), CAFFE_CUDA_NUM_THREADS>>>(
                        nc, (float *)(data + i*n1), (float)value);
                }
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }

    return tensor;
}


/////////////////////////////////////////////////////////////////////////////////////
static void ggml_compute_forward_get_rows_q(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst) {
}
static void ggml_compute_forward_get_rows_f16(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst) {
}
static void ggml_compute_forward_get_rows_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst) {
}

static void ggml_compute_forward_get_rows(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q8_0:
            {
                ggml_compute_forward_get_rows_q(params, src0, src1, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_get_rows_f16(params, src0, src1, dst);
            } break;
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_get_rows_f32(params, src0, src1, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}


























///////////////////////////////////////////////////////////////////////////////////////
void ggml_compute_forward_cuda(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    GGML_ASSERT(params);

    switch (tensor->op) {
        case GGML_OP_GET_ROWS:
            {
                ggml_compute_forward_get_rows(params, tensor->src0, tensor->src1, tensor);
            } break;
        default:
            GGML_ASSERT(false);
    }
}