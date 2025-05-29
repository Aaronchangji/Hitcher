#include <faiss/gpu/hitcher/gpu_profile.cuh>
#include <faiss/gpu/hitcher/gpu_worker.cuh>
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/InterleavedCodes.h>
#include <faiss/gpu/impl/RemapIndices.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <thrust/host_vector.h>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/impl/IVFAppend.cuh>
#include <faiss/gpu/impl/IVFFlat.cuh>
#include <faiss/gpu/impl/IVFFlatScan.cuh>
#include <faiss/gpu/impl/IVFInterleaved.cuh>
#include <faiss/gpu/impl/L2Norm.cuh>
#include <faiss/gpu/impl/Distance.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <faiss/gpu/utils/Transpose.cuh>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace faiss{

namespace hitcher {

DummyCache::DummyCache(int gpu_id, cudaStream_t stm, size_t num_vec, size_t d, faiss::gpu::GpuResources* resources) :
    device(gpu_id), scope(gpu_id), space(faiss::gpu::MemorySpace::Device), stream(stm),
    num_vec_page(num_vec), dim(d),
    d_cluster_sizes(
        resources,
        faiss::gpu::AllocInfo(faiss::gpu::AllocType::IVFLists,
            gpu_id,
            space,
            stream)
    ),
    d_cluster_ids_ptrs(
        resources,
        faiss::gpu::AllocInfo(faiss::gpu::AllocType::IVFLists,
            gpu_id,
            space,
            stream)
    ),
    d_cluster_codes_ptrs(
        resources,
        faiss::gpu::AllocInfo(faiss::gpu::AllocType::IVFLists,
            gpu_id,
            space,
            stream)
    )
{
    size_t avail_gpu_memory, total_gpu_memory;
    cudaMemGetInfo(&avail_gpu_memory, &total_gpu_memory);
    size_t cache_gpu_memory = size_t(avail_gpu_memory * kGPUMemForCacheRatio);
    size_t page_size = num_vec_page * sizeof(float) * dim;
    total_num_pages = cache_gpu_memory / page_size;
    cudaMalloc((void **)&d_cache_memory, page_size * total_num_pages);
    d_cluster_sizes.resize(total_num_pages, stream);
    d_cluster_codes_ptrs.resize(total_num_pages, stream);
    cudaMallocHost((void **)&h_pin_memory, page_size * total_num_pages);
    h_cluster_codes_ptrs.resize(total_num_pages);
    for (size_t i=0; i<total_num_pages; i++) {
        d_cluster_sizes.setAt(i, num_vec_page, stream);
        d_cluster_codes_ptrs.setAt(i, (void*)(d_cache_memory + i * page_size), stream);
        h_cluster_codes_ptrs[i] = h_pin_memory + i * page_size;
    }
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaStreamSynchronize(stream);
}

DummyCache::~DummyCache() {
    d_cluster_sizes.clear();
    d_cluster_codes_ptrs.clear();
    h_cluster_codes_ptrs.clear();
    cudaFree(d_cache_memory);
    cudaFreeHost(h_pin_memory);
}

DummyBatchInfo DummyCache::generateRandomBatch(int num_cluster, int num_query, int num_duplicated) {
    srand((unsigned int)(time(NULL)));
    DummyBatchInfo batch;
    batch.num_cluster = num_cluster;
    batch.num_query = num_query;
    size_t offset = 0;
    for (int i=0; i<num_cluster; i++) {
        batch.cids.push_back(rand() % total_num_pages);
        int num_query_for_cls = num_duplicated;
        batch.offsets.push_back(offset);
        offset += num_query_for_cls;
        for (int j=0; j<num_query_for_cls; j++) {
            batch.qids.push_back(rand() % num_query);
        }
    }
    batch.offsets.push_back(offset);
    return batch;
}

std::vector<int> DummyCache::generatePackVec(int total, int max, int size) {
    assert(total <= max * size);
    if (total == max * size) {
        return std::vector<int>(size, max);
    }

    // int num_query_per_cls = total / max;
    // std::vector<int> pack_vec(size, num_query_per_cls);
    // return pack_vec;

    std::vector<int> pack_vec(size, 1);
    int sum = size;
    std::random_device rd;
    std::mt19937 gen(rd());
    double mean = total / size;
    double lambda = 1.0 / mean;
    std::exponential_distribution<double> distribution(lambda);
    while (sum++ < total) {
        int pos = rand() % size;
        // int pos = static_cast<int>(distribution(gen)) % size;
        pack_vec[pos]++;
    }
    std::queue<int> not_fill_pos;
    for (int pos=0; pos<size; pos++) {
        if (pack_vec[pos] < max) {
            not_fill_pos.push(pos);
        }
    }
    for (int pos=0; pos<size; pos++) {
        while (pack_vec[pos] > max) {
            pack_vec[pos]--;
            int des = not_fill_pos.front();
            while (pack_vec[des] == max) {
                not_fill_pos.pop();
                des = not_fill_pos.front();
            }
            pack_vec[des]++;
        }
    }
    std::shuffle(pack_vec.begin(), pack_vec.end(), gen);
    int check_sum = 0;
    for (int pos=0; pos<size; pos++) {
        check_sum += pack_vec[pos];
    }
    assert(check_sum == total);
    return pack_vec;
}

DummyBatchInfo DummyCache::generateRandomBatch(int num_cluster, int num_query, int total_pack, int max_pack_per_cls, bool is_sort) {
    srand((unsigned int)(time(NULL)));
    // generate pack array
    std::vector<int> num_packs = generatePackVec(total_pack, max_pack_per_cls, num_cluster);
    if (is_sort) {
        std::sort(num_packs.begin(), num_packs.end(), std::greater<int>());
    }
    DummyBatchInfo batch;
    batch.num_cluster = num_cluster;
    batch.num_query = num_query;
    size_t offset = 0;
    for (int i=0; i<num_cluster; i++) {
        batch.cids.push_back(rand() % total_num_pages);
        int num_query_for_cls = num_packs[i];
        batch.offsets.push_back(offset);
        offset += num_query_for_cls;
        for (int j=0; j<num_query_for_cls; j++) {
            batch.qids.push_back(rand() % num_query);
        }
    }
    batch.offsets.push_back(offset);
    return batch;
}

DummyBatchInfo DummyCache::generateRandomBatchQC(int num_query, int total_num_query, int max_probe) {
    srand((unsigned int)(time(NULL)));

    DummyBatchInfo batch;
    batch.num_query = num_query;

    std::set<idx_t> selected_qids;
    for (int i = 0; i < num_query; i++) {
        // while (true) {
            // idx_t qid = rand() % total_num_query;
            // if (selected_qids.count(qid) == 0) {
            //     batch.qids.push_back(qid);
            //     selected_qids.emplace(qid);
            //     break;
            // }
        // }
        batch.qids.push_back(rand() % total_num_query);
        for (int j = 0; j < max_probe; j++) {
            idx_t cid = rand() % total_num_pages;
            batch.cids.push_back(cid);
        }
    }

    return batch;
}

double DummyCache::executeBatchQC(DummyBatchInfo &batch, faiss::gpu::GpuResources* resources, int max_probe) {
    // collect query vecs
    int num_queries = batch.num_query;
    faiss::gpu::DeviceTensor<float, 2, true> query_vecs(
        resources,
        faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream), 
        {num_queries, dim}
    );

    auto coarse_indices = faiss::gpu::toDeviceTemporary<int64_t, 2>(
        resources,
        device,
        (int64_t*)batch.cids.data(),
        stream,
        {num_queries, max_probe}
    );
    faiss::gpu::DeviceTensor<float, 2, true> distances(
        resources, 
        faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream), 
        {num_queries, 1}
    );
    faiss::gpu::DeviceTensor<idx_t, 2, true> indices(
        resources, 
        faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream), 
        {num_queries, 1}
    );
    cudaStreamSynchronize(stream);
    cudaEventRecord(start_event, stream);
    faiss::gpu::runIVFFlatScanHQC(
        query_vecs,
        coarse_indices,
        num_queries,
        d_cluster_codes_ptrs,
        d_cluster_ids_ptrs,
        d_cluster_sizes,
        num_vec_page,
        1,
        faiss::MetricType::METRIC_INNER_PRODUCT,
        distances,
        indices,
        resources,
        stream
    );
    cudaEventRecord(stop_event, stream);
    cudaEventSynchronize(stop_event);
    cudaStreamSynchronize(stream);
    float time_elapsed = 0;
    cudaEventElapsedTime(&time_elapsed, start_event, stop_event);
    return time_elapsed;  // ms
}

double DummyCache::executeBatch(DummyBatchInfo &batch, faiss::gpu::GpuResources* resources) {
    // collect query vecs
    faiss::gpu::DeviceTensor<float, 2, true> query_vecs(
        resources,
        faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream), 
        {batch.num_query, dim}
    );
    // search process, allocate gpu result buffer
    auto query_local_ids = faiss::gpu::toDeviceTemporary<int64_t, 1>(
        resources, 
        device,
        (int64_t*)batch.qids.data(), 
        stream,
        {int(batch.qids.size())}
    );
    auto query_offsets = faiss::gpu::toDeviceTemporary<int64_t, 1>(
        resources, 
        device,
        (int64_t*)batch.offsets.data(),
        stream,
        {int(batch.offsets.size())}
    );
    auto list_ids = faiss::gpu::toDeviceTemporary<int64_t, 1>(
        resources, 
        device,
        (int64_t*)batch.cids.data(),
        stream,
        {int(batch.cids.size())}
    );
    faiss::gpu::DeviceTensor<float, 2, true> distances(
        resources, 
        faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream), 
        {1, 1}
    );
    faiss::gpu::DeviceTensor<idx_t, 2, true> indices(
        resources, 
        faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream), 
        {1, 1}
    );
    cudaStreamSynchronize(stream);
    cudaEventRecord(start_event, stream);
    faiss::gpu::runIVFFlatScanHCC(
        query_vecs,
        query_local_ids,
        query_offsets,
        list_ids,
        d_cluster_codes_ptrs,
        d_cluster_ids_ptrs,
        d_cluster_sizes,
        num_vec_page,
        1,
        faiss::MetricType::METRIC_INNER_PRODUCT,
        distances,
        indices,
        resources,
        stream
    );
    cudaEventRecord(stop_event, stream);
    cudaEventSynchronize(stop_event);
    cudaStreamSynchronize(stream);
    float time_elapsed = 0;
    cudaEventElapsedTime(&time_elapsed, start_event, stop_event);
    // return time_elapsed/1000;
    return time_elapsed;  // ms
}

double DummyCache::transferBatch(int num_pages, faiss::gpu::GpuResources* resources) {
    size_t page_size = num_vec_page * sizeof(float) * dim;
    std::vector<int> page_ids;
    for (int i=0; i<num_pages; i++) {
        page_ids.push_back(rand() % total_num_pages);
    }
    cudaStreamSynchronize(stream);
    cudaEventRecord(start_event, stream);
    for (int i=0; i<num_pages; i++) {
        cudaMemcpyAsync(
            d_cache_memory + i * page_size,
            h_pin_memory + i * page_size,
            page_size,
            cudaMemcpyHostToDevice,
            stream
        );
        d_cluster_sizes.setAt(i, page_size, stream);
    }
    cudaEventRecord(stop_event, stream);
    cudaEventSynchronize(stop_event);
    cudaStreamSynchronize(stream);
    float time_elapsed = 0;
    cudaEventElapsedTime(&time_elapsed, start_event, stop_event);
    return time_elapsed/1000;
}

}

}
