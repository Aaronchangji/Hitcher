#ifndef HITCHER_STORAGE_H
#define HITCHER_STORAGE_H

#include <faiss/IndexIVFFlat.h>
#include <faiss/hitcher/common.h>
#include <faiss/hitcher/io.h>
#include <faiss/hitcher/task.h>
#include <unordered_map>
#include <omp.h>
#include <algorithm>
#include <atomic>
#include <deque>
#include <queue>
#include <set>
#include <faiss/hitcher/configure.h>

namespace faiss{

namespace hitcher {
    
struct Storage {
    faiss::IndexIVFFlat* index;
    idx_t nb, d;
    int8_t *vec_data;
    std::string dtype;
    size_t ele_bytes;
    std::vector<size_t> org_cluster_sizes;
    std::vector<std::vector<idx_t> > org_cluster_ids;
    std::vector<int8_t*> org_cluster_codes_ptrs;
    // org cls -> split cls
    std::unordered_map<idx_t, std::vector<idx_t> > id_map;
    std::vector<size_t> cluster_sizes;
    std::vector<idx_t*> cluster_ids_ptrs;
    std::vector<int8_t*> cluster_codes_ptrs;
    // std::vector<std::atomic<int>> cluster_access_cnt;
    std::deque<std::atomic<int>> cluster_access_cnt;
    size_t num_clusters;
    // query vectors
    int nq, dim;
    float *query_vecs_float32;
    int8_t *query_vecs_int8;
    std::vector<Query*> queries;
    std::atomic<int> num_finished_queries;
    FastConcurrentQueue<idx_t> top_ready_queries;
    FastConcurrentQueue<idx_t> finish_top_queries;
    FastConcurrentQueue<idx_t> bottom_ready_queries;
    FastConcurrentQueue<Task> finish_tasks;

    FastConcurrentQueue<TaskBatchStats> task_batch_stats_lists;
    std::atomic<bool> batch_finish_;  // For rummy test with batch-in&out

    Storage(Configure &config);
    ~Storage();
    void split(idx_t vec_bytes);
    float* getQueryVecFloat32(faiss::idx_t query_id);
    int8_t* getQueryVecInt8(faiss::idx_t query_id);
    size_t getClusterSize(faiss::idx_t cluster_id);
    int8_t* getClusterCodes(faiss::idx_t cluster_id);
    faiss::idx_t* getClusterIds(faiss::idx_t cluster_id);
    void addQuery(faiss::idx_t query_id);
    void getStats(int num_skip, int num_queries, int num_gpu = 0);
    void count_unique_access_cluster_rate(std::vector<faiss::idx_t>& query_batch);
};

using StoragePtr = std::shared_ptr<Storage>;

}

}

#endif