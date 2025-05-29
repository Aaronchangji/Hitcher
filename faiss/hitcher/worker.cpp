#include <faiss/hitcher/worker.h>

namespace faiss{

namespace hitcher {

Worker::Worker(WorkerType worker_type, StoragePtr storage_ptr, int k)
: worker_type_(worker_type), 
  storage_ptr_(storage_ptr), 
  k_(k), 
  nprobe_(storage_ptr->index->nprobe),
  num_running_task_(0),
  num_finish_task_(0),
  num_hit_task_(0) 
{}

Worker::~Worker() {
    storage_ptr_ = nullptr;
}

CPUWorker::CPUWorker(StoragePtr storage_ptr, int k, int num_thread, bool cpu_offload, std::vector<WorkerPtr>& gpu_workers)
: Worker(WorkerType::CPU, storage_ptr, k), gpu_workers_(gpu_workers), num_thread_(num_thread), cpu_offload_(cpu_offload), stop_(false) {
    if (gpu_workers_.size()) {
        no_gpu_ = false;
        num_thread_per_gpu_ = (num_thread + gpu_workers.size() - 1) / gpu_workers.size();
        stealed_tasks_.resize(num_thread_);
        for (auto & v : stealed_tasks_) {
            v.resize(gpu_workers_.size(), 0);
        }
    }
    threads_.resize(num_thread_);
    for (int i=0; i<num_thread_; i++) {
        threads_[i] = new std::thread(&faiss::hitcher::CPUWorker::thread_work, this, i);
    }
}

CPUWorker::~CPUWorker() {
    stop_ = true;
    int total = num_finish_task_;
    printf("[CPUWorker] cpu worker finish task number: %d\n", total);
    if (gpu_workers_.size()) {
        std::vector<int> num_tasks_from_gpu;
        num_tasks_from_gpu.resize(gpu_workers_.size(), 0);
        for (auto & th : stealed_tasks_) {
            for (int i = 0; i < th.size(); i++) {
                num_tasks_from_gpu[i] += th[i];
            }
        }
        for (int i = 0; i < num_tasks_from_gpu.size(); i++) {
            printf("[CPUWorker] CPU worker steal %d tasks from GPU %d.\n", num_tasks_from_gpu[i], i);
        }
    }
    fflush(stdout);
    for (int i=0; i<num_thread_; i++) {
        if (threads_[i] != nullptr) {
            threads_[i]->join();
            delete threads_[i];
            threads_[i] = nullptr;
        }
    }
    threads_.clear();
}

void CPUWorker::addTopTask(std::vector<faiss::idx_t> &qids) {
    num_running_task_ += qids.size();
    query_que_.enqueue_bulk(qids.begin(), qids.size());
}

void CPUWorker::addTopTask(faiss::idx_t qid) {
    num_running_task_++;
    query_que_.enqueue(qid);
}

void CPUWorker::addBottomTask(std::vector<Task> &tasks) {
    num_running_task_ += tasks.size();
    task_que_.enqueue_bulk(tasks.begin(), tasks.size());
}

void CPUWorker::addBottomTask(Task &task) {
    num_running_task_++;
    task_que_.enqueue(task);
}

void CPUWorker::searchTop(faiss::idx_t qid) {
    storage_ptr_->queries[qid]->top_execution_start_time = omp_get_wtime();
    const float *query_vec = storage_ptr_->getQueryVecFloat32(qid);
    int nprobe = storage_ptr_->index->nprobe;
    std::vector<faiss::idx_t> cluster_ids(nprobe);
    storage_ptr_->index->quantizer->assign(1, query_vec, cluster_ids.data(), nprobe);
    storage_ptr_->queries[qid]->cids = std::move(cluster_ids);
    storage_ptr_->queries[qid]->finish_top_time = omp_get_wtime();
    storage_ptr_->finish_top_queries.enqueue(qid);
}

float CPUWorker::computeDistance(const float *x, const float *y, int dim) {
    float dis = (storage_ptr_->index->metric_type == METRIC_INNER_PRODUCT)
                ? faiss::fvec_inner_product(x, y, dim)
                : faiss::fvec_L2sqr(x, y, dim);
    return dis;
}

void CPUWorker::searchBottom(Task &task) {
    size_t cluster_size = storage_ptr_->getClusterSize(task.cluster_id);
    faiss::idx_t *cluster_ids = storage_ptr_->getClusterIds(task.cluster_id);
    faiss::MetricType metric_type = storage_ptr_->index->metric_type;
    // faiss::hitcher::PartialResult result;
    // result.task = task;
    // std::vector<float> &distances = result.distances;
    // distances.resize(k_);
    // std::vector<faiss::idx_t> &indices = result.result_indices;
    // indices.resize(k_);
    // std::vector<float> distances(k_);
    // std::vector<faiss::idx_t> indices(k_);
    // using HeapForIP = faiss::CMin<float, idx_t>;
    // using HeapForL2 = faiss::CMax<float, idx_t>;
    // auto init_result = [&](float* simi, faiss::idx_t* idxi) {
    //     if (metric_type == METRIC_INNER_PRODUCT) {
    //         heap_heapify<HeapForIP>(k_, simi, idxi);
    //     } else {
    //         heap_heapify<HeapForL2>(k_, simi, idxi);
    //     }
    // };
    // init_result(distances.data(), indices.data());
    // auto reorder_result = [&](float* simi, idx_t* idxi) {
    //     if (metric_type == METRIC_INNER_PRODUCT) {
    //         heap_reorder<HeapForIP>(k_, simi, idxi);
    //     } else {
    //         heap_reorder<HeapForL2>(k_, simi, idxi);
    //     }
    // };
    int dim = 0;
    float *query_vec = nullptr;
    float *cluster_vecs = nullptr;
    if (storage_ptr_->dtype == "float32") {
        dim = storage_ptr_->index->d;
        query_vec = storage_ptr_->getQueryVecFloat32(task.query_id);
        cluster_vecs = (float*)storage_ptr_->getClusterCodes(task.cluster_id);
    } else {
        dim = storage_ptr_->index->d / 4;
        query_vec = (float*)storage_ptr_->getQueryVecInt8(task.query_id);
        cluster_vecs = (float*)storage_ptr_->getClusterCodes(task.cluster_id);
    }
    std::vector<float> computed_distances(cluster_size);
    for (size_t i=0; i<cluster_size; i++) {
        computed_distances[i] = computeDistance(cluster_vecs, query_vec, dim);
        cluster_vecs += dim;
    }
    std::vector<float> distances(k_);
    std::vector<idx_t> indices(k_);
    using HeapForIP = faiss::CMin<float, idx_t>;
    using HeapForL2 = faiss::CMax<float, idx_t>;
    auto init_result = [&](float* simi, faiss::idx_t* idxi) {
        if (metric_type == METRIC_INNER_PRODUCT) {
            heap_heapify<HeapForIP>(k_, simi, idxi);
        } else {
            heap_heapify<HeapForL2>(k_, simi, idxi);
        }
    };
    init_result(distances.data(), indices.data());
    for (idx_t i=0; i<cluster_size; i++) {
        float dis = computed_distances[i];
        if (dis < distances[0]) {
            faiss::maxheap_replace_top(k_, distances.data(), indices.data(), dis, i);
        }
    }
    // for (size_t i=0; i<cluster_size; i++) {
    //     float dis = computeDistance(cluster_vecs, query_vec, dim);
    //     if (dis < distances[0]) {
    //         int64_t id = cluster_ids[i];
    //         faiss::maxheap_replace_top(k_, distances.data(), indices.data(), dis, id);
    //     }
    //     cluster_vecs += dim;
    // }
    // reorder_result(distances.data(), indices.data());
    // storage_ptr_->addQueryResult(task.query_id, result);
    // storage_ptr_->addPartialResult(result);
    storage_ptr_->finish_tasks.enqueue(task);
}

bool CPUWorker::steal_from_gpu(Task& task, int thread_id) {
    int assigned_gpu_id = thread_id / num_thread_per_gpu_;
    bool success = gpu_workers_[assigned_gpu_id]->stealTaskFrom(task);

    if (success) {
        stealed_tasks_[thread_id][assigned_gpu_id]++;
    }

    // TBD: If not, there is a chance to steal from other gpus?

    return success;
}

void CPUWorker::thread_work(int thread_id) {
    // bindCore(thread_id);
    while (!stop_) {
        idx_t qid;
        Task task;
        if (task_que_.try_dequeue(task)) {
            searchBottom(task);
            num_running_task_--;
            num_finish_task_++;
        } else if (query_que_.try_dequeue(qid)) {
            searchTop(qid);
            num_running_task_--;
        // } else if (master_ != nullptr && master_->stealTaskFrom(task)) {
        } else if (!no_gpu_ && cpu_offload_ && steal_from_gpu(task, thread_id)) {
            searchBottom(task);
            num_finish_task_++;
        }
        // else if (storage_ptr_->top_ready_queries.try_dequeue(qid)) {
        //     searchTop(qid);
        // }
    }
}


}

}