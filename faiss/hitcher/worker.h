#ifndef HITCHER_WORKER_H
#define HITCHER_WORKER_H

#include <faiss/hitcher/task.h>
#include <faiss/hitcher/storage.h>
#include <faiss/hitcher/common.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/Heap.h>
#include <thread>
#include <assert.h>

namespace faiss{

namespace hitcher {

enum WorkerType {CPU, GPU};

static std::vector<std::string> WorkerTypeStr {"CPU", "GPU"};

class Worker;

using WorkerPtr = std::shared_ptr<Worker>;

class Worker {
public:
    WorkerType worker_type_;
    int k_, nprobe_;
    StoragePtr storage_ptr_;
    std::atomic<int> num_running_task_;
    std::atomic<int> num_finish_task_;
    int num_hit_task_;

public:
    Worker(WorkerType worker_type, StoragePtr storage_ptr, int k);
    virtual void launchWorker() = 0;
    virtual ~Worker();
    virtual void addTopTask(std::vector<faiss::idx_t> &qids) = 0;
    virtual void addTopTask(faiss::idx_t qid) = 0;
    virtual void addBottomTask(std::vector<Task> &tasks) = 0;
    virtual void addBottomTask(Task &task) = 0;
    virtual bool stealTaskFrom(Task &task) {return false;};
    virtual void getStats() { return; };
};


class CPUWorker : public Worker {
private:
    int num_thread_, num_thread_per_gpu_;
    std::vector<WorkerPtr>& gpu_workers_;
    std::vector<std::thread*> threads_;
    faiss::hitcher::FastConcurrentQueue<faiss::idx_t> query_que_;
    faiss::hitcher::FastConcurrentQueue<Task> task_que_;
    std::vector<std::vector<int>> stealed_tasks_;
    std::atomic<bool> stop_;
    bool cpu_offload_{true};
    bool no_gpu_{true};

public:
    CPUWorker(StoragePtr storage_ptr, int k, int num_thread, bool cpu_offload, std::vector<WorkerPtr>& gpu_workers_);
    virtual void launchWorker() {return;};
    virtual ~CPUWorker();
    virtual void addTopTask(std::vector<faiss::idx_t> &qids);
    virtual void addTopTask(faiss::idx_t qid);
    virtual void addBottomTask(std::vector<Task> &tasks);
    virtual void addBottomTask(Task &task);
    inline void searchTop(faiss::idx_t qid);
    inline float computeDistance(const float *x, const float *y, int dim);
    inline void searchBottom(Task &task);
    bool steal_from_gpu(Task &task, int thread_id);
    void thread_work(int thread_id);

};

}

}

#endif