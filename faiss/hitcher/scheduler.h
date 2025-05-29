#ifndef HITCHER_SCHEDULER_H
#define HITCHER_SCHEDULER_H

#include <faiss/hitcher/worker.h>
#include <random>
#include <deque>
#include <unordered_map>

namespace faiss{

namespace hitcher {

class Scheduler {
private:
    StoragePtr storage_ptr_;
    WorkerPtr cpu_worker_;
    std::vector<WorkerPtr>& gpu_workers_;
    // WorkerPtr master_worker_, slave_worker_;
    bool no_gpu_{true};
    std::atomic<bool> stop_;
    std::thread *scheduler_;
    size_t top_search_counter_{0}, num_cls_per_gpu_;
    bool is_rummy_{false};

public:
    Scheduler(StoragePtr storage_ptr, WorkerPtr cpu_worker, std::vector<WorkerPtr>& gpu_workers, bool is_rummy);
    ~Scheduler();
    void thread_work();
    int get_gpu_id(idx_t cls_id);
    WorkerPtr get_top_search_worker();
    void assign_bottom_tasks(std::vector<Task>& tasks);
};


}

}

#endif