#ifndef HITCHER_GPU_UTILS_H
#define HITCHER_GPU_UTILS_H

#include <cuda_runtime.h>
#include <vector>

namespace faiss{

namespace hitcher {

void gather(int64_t* src_address, int num, int dim, float *dst, cudaStream_t stream);

void gather(float *data, int64_t *offsets, int num, int dim, float *dst, cudaStream_t stream);



}

}

#endif