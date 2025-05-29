#ifndef HITCHER_REDUCER_H
#define HITCHER_REDUCER_H

#include <faiss/hitcher/worker.h>
#include <unordered_set>

namespace faiss{

namespace hitcher {
    
class Reducer {
private:
    int k_;
    StoragePtr storage_ptr_;
    int num_thread_;
    std::vector<std::thread*> threads_;
    std::atomic<bool> stop_;
    bool is_rummy_{false};

public:
    Reducer(StoragePtr storage_ptr, int k, int num_thread, bool is_rummy);
    ~Reducer();
    inline int getNumValid(idx_t cid);
    inline void reduce(PartialResult &partial_result, float *distances, faiss::idx_t *indices);
    inline void classifyBatchResult(BatchPartialResult &partial_result);
    void thread_work(int thread_id);
    
};

}

}

#endif