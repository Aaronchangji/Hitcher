#ifndef HITCHER_GPU_PROFILE_H
#define HITCHER_GPU_PROFILE_H

#include <faiss/hitcher/worker.h>
#include <faiss/gpu/hitcher/h_gpu_utils.cuh>
#include <faiss/Clustering.h>
#include <faiss/IndexIVF.h>
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <faiss/Index.h>
#include <queue>
#include <faiss/hitcher/storage.h>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/DeviceVector.cuh>
#include <faiss/gpu/StandardGpuResources.h>
#include <unordered_map>
#include <map>
#include <set>
#include <unordered_set>
#include <iterator>
#include <list>
#include <cstdio>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <random>
#include <queue>
#include <algorithm>
#include <functional>

namespace faiss{

namespace hitcher {

struct DummyBatchInfo {
    int num_cluster, num_query;
    std::vector<idx_t> cids;
    std::vector<idx_t> offsets;
    std::vector<idx_t> qids;
};

struct DummyCache {
    std::default_random_engine random_engine;
    int device;
    idx_t num_vec_page;  // Number of vectors per page
    idx_t dim;
    cudaEvent_t start_event, stop_event;
    cudaStream_t stream;
    faiss::gpu::DeviceScope scope;
    faiss::gpu::MemorySpace space;
    faiss::gpu::DeviceVector<idx_t> d_cluster_sizes;
    faiss::gpu::DeviceVector<void*> d_cluster_ids_ptrs;
    faiss::gpu::DeviceVector<void*> d_cluster_codes_ptrs;
    uint8_t *d_cache_memory;
    size_t total_num_pages;
    uint8_t *h_pin_memory;
    std::vector<uint8_t*> h_cluster_codes_ptrs;

    DummyCache(int gpu_id, cudaStream_t stm, size_t num_vec, size_t d, faiss::gpu::GpuResources* resources);
    ~DummyCache();
    DummyBatchInfo generateRandomBatch(int num_cluster, int num_query, int num_duplicated);
    inline std::vector<int> generatePackVec(int total, int max, int size);
    DummyBatchInfo generateRandomBatch(int num_cluster, int num_query, int total_pack, int max_pack_per_cls, bool is_sort);
    DummyBatchInfo generateRandomBatchQC(int num_query, int total_num_query, int max_probe);
    double executeBatch(DummyBatchInfo &batch, faiss::gpu::GpuResources* resources);
    double executeBatchQC(DummyBatchInfo &batch, faiss::gpu::GpuResources* resources, int max_probe);
    double transferBatch(int num_pages, faiss::gpu::GpuResources* resources);
};

}

}

#endif