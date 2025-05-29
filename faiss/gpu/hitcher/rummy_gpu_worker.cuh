#ifndef HITCHER_RUMMY_GPU_WORKER_H
#define HITCHER_RUMMY_GPU_WORKER_H

#include <faiss/gpu/hitcher/gpu_worker.cuh>

namespace faiss{

namespace hitcher {

class RummyGPUWorker : public GPUWorker {
public:
    std::atomic<int> num_task_to_dispatch_;
    std::atomic<int> num_running_tasks_;
    faiss::hitcher::FastConcurrentQueue<idx_t> ready_cids_;

public:
    RummyGPUWorker(StoragePtr storage_ptr, int k, int gpu_id, int max_query_batch_size, int max_bottom_batch_size, int max_pack_num, std::string kernel_mode, int transfer_batch_size);
    ~RummyGPUWorker();
    virtual bool stealTaskFrom(Task &task);
    // search
    virtual void makeBatchWork();
    inline void syncSearchTop2(std::vector<idx_t> &query_batch);
    inline void selectBatch(BatchInfo &batch);
    virtual void searchWork();
    // dispatch
    virtual void dispatchWork();
    
};

}

}

#endif