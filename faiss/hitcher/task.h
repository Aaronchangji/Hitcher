#ifndef HITCHER_TASK_H
#define HITCHER_TASK_H

#include <faiss/MetricType.h>
#include <memory>
#include <atomic>
#include <vector>
#include "omp.h"
#include <cuda_runtime.h>
#include <faiss/hitcher/common.h>

namespace faiss{

namespace hitcher {

struct Task {
    faiss::idx_t query_id;
    faiss::idx_t cluster_id;
    int gpu_id{-1};
    bool wait_transfer{false};
    Task() {};
    Task(faiss::idx_t qid, faiss::idx_t cid) : query_id(qid), cluster_id(cid) {};
    void set_gpu_id(int _gpu_id) { gpu_id = _gpu_id; }
    void set_wait_transfer() { wait_transfer = true; }

    bool operator==(const Task& other) const {
        return cluster_id == other.cluster_id && query_id == other.query_id && gpu_id == other.gpu_id;
    }
};

struct Query {
    faiss::idx_t query_id;
    double arrival_time;
    double top_execution_start_time;
    double finish_top_time;
    double dispatch_time;
    // double finish_bottom_time;
    double finish_time;
    std::atomic<int> remain_cluster;
    std::vector<float> distances;
    std::vector<faiss::idx_t> result_indices;
    std::vector<faiss::idx_t> cids;
    std::vector<Task> tasks;
    int top_search_gpu_id;
};

struct QueryBatch {
    int batch_id;
    double start_time;
    double finish_time;
    std::atomic<int> remain_top_task;
    std::atomic<int> remain_bottom_task;
    std::vector<faiss::idx_t> query_ids;
};

struct TaskBatchStats {
    int num_cls;
    int num_queries;
    int num_unique_queries;
    int num_unique_cls;
};

struct PartialResult {
    Task task;
    std::vector<float> distances;
    std::vector<faiss::idx_t> result_indices;
};

struct BatchPartialResult {
    int batch_size;
    std::vector<Task> tasks;
    std::vector<float> distances;
    std::vector<faiss::idx_t> result_indices;
};

struct Stats {
    int num_queries;
    double p50_latency;
    double p90_latency;
    double p95_latency;
    double p99_latency;

    Stats() {
        num_queries = 0;
        p50_latency = 0;
        p90_latency = 0;
        p95_latency = 0;
        p99_latency = 0;
    }

    void show() {
        printf("complete %d queries\n", num_queries);
        printf("P50 latency: %.3lf ms\n", p50_latency * 1000);
        printf("P90 latency: %.3lf ms\n", p90_latency * 1000);
        printf("P95 latency: %.3lf ms\n", p95_latency * 1000);
        printf("P99 latency: %.3lf ms\n", p99_latency * 1000);
    }

};

}

}

#endif