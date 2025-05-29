#ifndef HITCHER_CPU_PROFILE_H
#define HITCHER_CPU_PROFILE_H

#include <faiss/hitcher/worker.h>
#include <random>

namespace faiss{

namespace hitcher {

struct DummyStorage {
    std::default_random_engine random_engine;
    idx_t num_vec_page;
    idx_t dim;
    size_t total_num_pages;
    std::vector<idx_t> id_data;
    float *pages_data;
    faiss::MetricType metric_type;
    int topk;
    faiss::hitcher::FastConcurrentQueue<Task> task_que;
    std::atomic<int> num_finish;
    std::atomic<bool> stop;
    std::vector<std::thread*> workers;

    DummyStorage(size_t num_vec, size_t d, int k, faiss::MetricType metric);
    ~DummyStorage();
    float computeDistance(const float *x, const float *y);
    void searchBottom(int thread_id);
    void setThreadNum(int num_thread);
    double profile(int num_task);
};

}

}

#endif