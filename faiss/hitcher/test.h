#ifndef HITCHER_TEST_H
#define HITCHER_TEST_H

#include <iostream>
#include <string>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/Heap.h>

namespace faiss{

namespace hitcher {
    void loadIndex(std::string index_file);

    float computeDistance(const float *x, 
                          const float *y, 
                          size_t d, 
                          faiss::MetricType metric_type) {
        float dis = (metric_type == METRIC_INNER_PRODUCT)
                    ? faiss::fvec_inner_product(x, y, d)
                    : faiss::fvec_L2sqr(x, y, d);
        return dis;
    }

    float computeDistanceGPU(const float *x, 
                            const float *y, 
                            size_t d, 
                            faiss::MetricType metric_type) {
        float dis = (metric_type == METRIC_INNER_PRODUCT)
                    ? faiss::fvec_inner_product(x, y, d)
                    : faiss::fvec_L2sqr(x, y, d);
        return dis;
    }

    void searchBottom(float *query_vec, 
                        float *cluster_vecs, 
                        size_t cluster_size, 
                        faiss::idx_t *cluster_ids,
                        int k,
                        size_t d,
                        faiss::MetricType metric_type,
                        float *distances,
                        faiss::idx_t *indices) {
        using HeapForIP = faiss::CMin<float, idx_t>;
        using HeapForL2 = faiss::CMax<float, idx_t>;
        auto init_result = [&](float* simi, faiss::idx_t* idxi) {
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_heapify<HeapForIP>(k, simi, idxi);
            } else {
                heap_heapify<HeapForL2>(k, simi, idxi);
            }
        };
        auto reorder_result = [&](float* simi, idx_t* idxi) {
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_reorder<HeapForIP>(k, simi, idxi);
            } else {
                heap_reorder<HeapForL2>(k, simi, idxi);
            }
        };
        init_result(distances, indices);
        for (size_t i=0; i<cluster_size; i++) {
            float dis = computeDistance(cluster_vecs, query_vec, d, metric_type);
            if (dis < distances[0]) {
                int64_t id = cluster_ids[i];
                faiss::maxheap_replace_top(k, distances, indices, dis, id);
            }
            cluster_vecs += d;
        }
        reorder_result(distances, indices);
    }
}

}

#endif