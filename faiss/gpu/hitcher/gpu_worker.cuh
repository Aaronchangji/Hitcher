#ifndef HITCHER_GPU_WORKER_H
#define HITCHER_GPU_WORKER_H

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
#include <chrono>
#include <fstream>
#include <optional>

namespace faiss{

namespace hitcher {

struct Cache {
    StoragePtr storage_ptr;
    int gpu_id;
    faiss::gpu::MemorySpace space;
    faiss::gpu::DeviceTensor<float, 2, true> d_centroids;
    faiss::gpu::DeviceTensor<float, 1, true> d_centroids_norms;
    faiss::gpu::DeviceVector<idx_t> d_cluster_sizes;
    faiss::gpu::DeviceVector<void*> d_cluster_ids_ptrs; //empty
    int ele_bytes;
    faiss::gpu::DeviceVector<void*> d_cluster_codes_ptrs;
    uint8_t *d_cache_memory;
    std::vector<uint8_t*> d_cluster_data;
    int total_num_pages;
    std::queue<idx_t> free_pages;
    // LFU cache
    std::vector<idx_t> cid2pid, pid2cid;
    std::vector<int> cluster_access_cnt;
    std::vector<int> cluster_pin_cnt;
    std::map<int, std::list<idx_t> > freq2nodelist;
    std::vector<std::list<idx_t>::iterator> cid2node;

    Cache(StoragePtr storage_pt, faiss::gpu::GpuResources* resources, int gpu_id, cudaStream_t stream);
    ~Cache();
    void init(cudaStream_t stream);
    inline bool hit(int cid);
    inline void pin(int cid);
    inline void unpin(int cid);
    inline bool canEvit();
    inline idx_t evit();
    inline void pinEvit(int cid);
};

struct ExGPUResources {
    int device;
    faiss::gpu::StandardGpuResources top_res, bottom_res;
    // cudaStream_t compute_stream; use the default stream in res
    cudaStream_t top_stream, bottom_stream, transfer_stream;
    std::queue<cudaEvent_t> event_pool;
    ExGPUResources(int gpu_id, StoragePtr storage_ptr);
    ~ExGPUResources();
};

struct BatchInfo {
    // for cc flat csr
    // for qc cid is two dimensional, clustered by qid, padding by -1
    int num_queries, num_cls;
    std::vector<idx_t> cids;
    std::vector<idx_t> offsets;
    std::vector<idx_t> qids;
    std::vector<float*> unique_query_address;
    int unique_query_cnt;
    std::vector<Task> tasks;
};

enum KernelMode {QC, CC};

// This operator should return true if the first argument should come after the second argument in the priority queue.
struct TaskCmp {
  bool operator() (const Task &a, const Task &b) {
    return a.query_id > b.query_id;
  }
};


class GPUWorker : public Worker {
public:
    ExGPUResources gpu_res_;
    Cache *gpu_cache_;
    int gpu_id_;
    // dispatch threads
    faiss::hitcher::FastConcurrentQueue<faiss::idx_t> query_que_;
    faiss::hitcher::FastConcurrentQueue<Task> task_que_;
    faiss::hitcher::FastConcurrentQueue<Task> wait_page_task_que_;
    faiss::hitcher::FastConcurrentQueue<Task> miss_task_que_;
    std::vector<std::queue<Task> > wait_transfer_ques_;
    faiss::hitcher::FastConcurrentQueue<Task> finish_transfer_task_que_;
    faiss::hitcher::FastConcurrentQueue<Task> finish_search_task_que_;
    // transfer threads
    int max_transfer_batch_size_;
    std::atomic<int> num_transfering_task_;
    faiss::hitcher::FastConcurrentQueue<Task> transfer_task_que_;
    std::queue<Task> pending_que_;
    std::queue<Task> transfering_que_;
    std::queue<cudaEvent_t> transfer_event_queue_;
    // search threads
    // faiss::hitcher::ConcurrentPriorityQueue<Task, TaskCmp> ready_tasks_;
    faiss::hitcher::FastConcurrentQueue<Task> ready_tasks_;     // Tasks hit the cache in GPU
    faiss::hitcher::FastConcurrentQueue<Task> transfered_ready_tasks_;     // Tasks finish transfering
    std::optional<Task> peek_ready_task_, peek_transfered_ready_task_;
    faiss::hitcher::FastConcurrentQueue<Task> ready_task_que_per_cls_;
    faiss::hitcher::FastConcurrentQueue<Task> transfered_ready_task_que_per_cls_;
    std::vector<int> hitch_ride_cnt_per_cls_ready_task_;
    std::vector<int> hitch_ride_cnt_per_cls_transfer_ready_task_;
    std::vector<moodycamel::ProducerToken> ptoks_for_ready_tasks_;
    std::vector<moodycamel::ProducerToken> ptoks_for_transfered_ready_tasks_;
    std::vector<std::optional<Task>> peek_ready_task_per_cls_;
    std::vector<std::optional<Task>> peek_transfered_ready_task_per_cls_;

    int select_from_transfer_queue{0};
    int hitch_ride_from_transfer_queue{0};
    int select_from_ready_queue{0};
    int hitch_ride_from_ready_queue{0};

    std::atomic<int> num_ready_batch_;
    faiss::hitcher::FastConcurrentQueue<BatchInfo> ready_batch_;
    double top_batch_threshold_;  // in ms
    int max_top_batch_size_;
    int max_bottom_batch_size_;
    int max_pack_num_;
    uint8_t *pin_top_buffer_, *pin_bottom_buffer_;
    float *pin_bottom_distances_buffer_;
    idx_t *pin_top_indices_buffer_, *pin_bottom_indices_buffer_;

    std::vector<double> make_batch_times;
    std::vector<double> get_ready_task_times;
    std::vector<double> get_hitch_task_times;
    // workers
    std::thread *batch_maker_;
    std::thread *search_thread_;
    std::thread *transfer_thread_;
    std::thread *dispatch_thread_;
    std::atomic<bool> stop_;
    KernelMode kernel_mode_;

    // debug
    // std::atomic<int> num_finished_top_query{0};
    // std::atomic<int> num_launched_bottom_task{0};
    // std::atomic<int> num_finished_bottom_task{0};
    // std::atomic<int> num_added_ready_task{0};
    // std::atomic<int> num_added_transfered_ready_task{0};

public:
    GPUWorker(StoragePtr storage_ptr, int k, int gpu_id, int max_query_batch_size, int max_bottom_batch_size, int max_pack_num, double top_batch_threshold, std::string kernel_mode, int transfer_batch_size);
    void initWork(std::vector<moodycamel::ProducerToken> *token_que1, std::vector<moodycamel::ProducerToken> *token_que2, int num_token_per_thread);
    virtual void launchWorker();
    virtual ~GPUWorker();
    virtual void addTopTask(std::vector<faiss::idx_t> &qids);
    virtual void addTopTask(faiss::idx_t qid);
    virtual void addBottomTask(std::vector<Task> &tasks);
    virtual void addBottomTask(Task &task);
    virtual bool stealTaskFrom(Task &task);
    // search
    inline void searchTop(std::vector<idx_t> &query_batch);
    inline void syncSearchTop(std::vector<idx_t> &query_batch);
    inline void searchBottomCC(BatchInfo &batch);
    inline void searchBottomQC(BatchInfo &batch); 
    inline bool getHitchRideTask(Task& task, faiss::idx_t cluster_id);
    inline bool getReadyTask(Task& task);
    void removeTaskFromClsQues(const Task task);
    inline void selectCC(BatchInfo &batch);
    inline void selectQC(BatchInfo &batch);
    inline void syncSearchBottom(BatchInfo &batch);
    virtual void searchWork();
    virtual void makeBatchWork();
    // transfer
    inline void transfer(Task &task, cudaEvent_t event);
    virtual void transferWork();
    // dispatch
    inline void addReadyTask(Task &task, bool after_transfered);
    inline void addReadyTask(std::vector<Task> &task, bool after_transfered);
    virtual void dispatchWork();

    // stats
    virtual void getStats();
};


class RummyGPUWorker : public GPUWorker {
public:
    std::atomic<int> num_batch_task_to_dispatch_;
    std::atomic<int> num_batch_task_remain_;
    faiss::hitcher::FastConcurrentQueue<idx_t> ready_cids_;

public:
    RummyGPUWorker(StoragePtr storage_ptr, int k, int gpu_id, int max_query_batch_size, int max_bottom_batch_size, int max_pack_num, double top_batch_threshold, std::string kernel_mode, int transfer_batch_size);
    // virtual void launchWorker();
    virtual ~RummyGPUWorker();
    virtual bool stealTaskFrom(Task &task);
    // search
    // virtual void makeBatchWork();
    void makeBatchWork() override;
    inline void syncSearchTop2(std::vector<idx_t> &query_batch);
    inline void selectBatch(BatchInfo &batch);
    // virtual void searchWork();
    void searchWork() override;
    // dispatch
    // virtual void dispatchWork();
    void dispatchWork() override;
    
};


}

}

#endif