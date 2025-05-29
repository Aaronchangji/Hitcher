#include <faiss/gpu/hitcher/h_gpu_utils.cuh>

namespace faiss{

namespace hitcher {

__global__ 
void GatherKernel(int64_t *srcs_addresses, float *dst, int num, int dim) {
  int out_row = blockIdx.x * blockDim.y + threadIdx.y;
  int stride = blockDim.y * gridDim.x;
  while (out_row < num) {
    float *input = (float*)(srcs_addresses[out_row]);
    int offset = out_row * dim;
    int idx = threadIdx.x;
    while (idx < dim) {
      dst[offset + idx] = input[idx];
      idx += blockDim.x;
    }
    out_row += stride;
  }
}

__global__ 
void GatherKernel(float *data, int64_t *offsets, float *dst, int num, int dim) {
  int out_row = blockIdx.x * blockDim.y + threadIdx.y;
  int stride = blockDim.y * gridDim.x;
  while (out_row < num) {
    float *input = data + offsets[out_row] * dim;
    int offset = out_row * dim;
    int idx = threadIdx.x;
    while (idx < dim) {
      dst[offset + idx] = input[idx];
      idx += blockDim.x;
    }
    out_row += stride;
  }
}

void gather(int64_t* src_address, int num, int dim, float *dst, cudaStream_t stream) {
    dim3 block(128, 1);
    while (static_cast<int64_t>(block.x) > dim) {
        block.x /= 2;
        block.y *= 2;
    }
    const dim3 grid((num + block.y - 1) / block.y);
    GatherKernel<<<grid, block, 0, stream>>>(src_address, dst, num, dim);
}

void gather(float *data, int64_t *offsets, int num, int dim, float *dst, cudaStream_t stream) {
    dim3 block(128, 1);
    while (static_cast<int64_t>(block.x) > dim) {
        block.x /= 2;
        block.y *= 2;
    }
    const dim3 grid((num + block.y - 1) / block.y);
    GatherKernel<<<grid, block, 0, stream>>>(data, offsets, dst, num, dim);
}

}

}