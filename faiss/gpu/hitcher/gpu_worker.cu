#include <faiss/gpu/hitcher/gpu_worker.cuh>
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/InterleavedCodes.h>
#include <faiss/gpu/impl/RemapIndices.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <thrust/host_vector.h>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/impl/IVFAppend.cuh>
#include <faiss/gpu/impl/IVFFlat.cuh>
#include <faiss/gpu/impl/IVFFlatScan.cuh>
#include <faiss/gpu/impl/IVFInterleaved.cuh>
#include <faiss/gpu/impl/L2Norm.cuh>
#include <faiss/gpu/impl/Distance.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <faiss/gpu/utils/Transpose.cuh>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <omp.h>

namespace faiss{

namespace hitcher {

Cache::Cache(StoragePtr storage_pt, faiss::gpu::GpuResources* resources, int gpu_id, cudaStream_t stream)
:   storage_ptr(storage_pt),
    gpu_id(gpu_id),
    space(faiss::gpu::MemorySpace::Device),
    d_centroids(
        resources,
        faiss::gpu::AllocInfo(
            faiss::gpu::AllocType::FlatData,
            gpu_id,
            space,
            stream
        ),
        {int(storage_ptr->index->nlist), storage_ptr->dim}
    ),
    d_centroids_norms(
        resources,
        faiss::gpu::AllocInfo(
            faiss::gpu::AllocType::FlatData,
            gpu_id,
            space,
            stream
        ),
        {int(storage_ptr->index->nlist)}
    ),
    d_cluster_sizes(
        resources,
        faiss::gpu::AllocInfo(faiss::gpu::AllocType::IVFLists,
            gpu_id,
            space,
            stream)
    ),
    d_cluster_ids_ptrs(
        resources,
        faiss::gpu::AllocInfo(faiss::gpu::AllocType::IVFLists,
            gpu_id,
            space,
            stream)
    ),
    d_cluster_codes_ptrs(
        resources,
        faiss::gpu::AllocInfo(faiss::gpu::AllocType::IVFLists,
            gpu_id,
            space,
            stream)
    )
{
    init(stream);
}

Cache::~Cache() {
    d_cluster_sizes.clear();
    d_cluster_ids_ptrs.clear();
    d_cluster_codes_ptrs.clear();
    cudaFree(d_cache_memory);
    d_cluster_data.clear();
}

void Cache::init(cudaStream_t stream) {
    faiss::gpu::DeviceScope scope(gpu_id);
    // init top index
    faiss::IndexFlat* top_index = dynamic_cast<faiss::IndexFlat*>(storage_ptr->index->quantizer);
    cudaMemcpyAsync(
        d_centroids.data(),
        top_index->get_xb(),
        storage_ptr->index->nlist * storage_ptr->dim * sizeof(float),
        cudaMemcpyHostToDevice,
        stream
    );
    faiss::gpu::runL2Norm(d_centroids, true, d_centroids_norms, true, stream);
    cudaStreamSynchronize(stream);
    // init bottom index
    size_t avail_gpu_memory;
    size_t total_gpu_memory;
    cudaMemGetInfo(&avail_gpu_memory, &total_gpu_memory);
    printf("total mem: %ld, avail mem: %ld\n", total_gpu_memory, avail_gpu_memory);
    size_t cache_gpu_memory = size_t(avail_gpu_memory * kGPUMemForCacheRatio);
    ele_bytes = (storage_ptr->dtype == "float32") ? 4 : 1;
    size_t page_size = kNumVecPerCluster * ele_bytes * storage_ptr->dim;
    size_t num_page = cache_gpu_memory / page_size;
    total_num_pages = num_page;
    cudaMalloc((void **)&d_cache_memory, page_size * num_page);
    d_cluster_sizes.resize(num_page, stream);
    d_cluster_sizes.setAll(0, stream);
    d_cluster_codes_ptrs.resize(num_page, stream);
    d_cluster_data.resize(num_page, nullptr);
    for (int i=0; i<num_page; i++) {
        d_cluster_codes_ptrs.setAt(i, (void*)(d_cache_memory + i * page_size), stream);
        d_cluster_data[i] = d_cache_memory + i * page_size;
        free_pages.push(i);
    }
    cudaStreamSynchronize(stream);
    size_t num_cluster = storage_ptr->cluster_sizes.size();
    cid2pid.resize(num_cluster, -1);
    pid2cid.resize(num_page, -1);
    cluster_access_cnt.resize(num_cluster, 0);
    cluster_pin_cnt.resize(num_cluster, 0);
    cid2node.resize(num_cluster);
    cudaStreamSynchronize(stream);
    printf("num clus: %ld, num page: %ld\n", num_cluster, num_page);
}

bool Cache::hit(int cid) {
    return (cid2pid[cid] >= 0);
}

void Cache::pin(int cid) {
    if (cluster_pin_cnt[cid]++ == 0) {
        int fc = cluster_access_cnt[cid];
        freq2nodelist[fc].erase(cid2node[cid]);
        if (freq2nodelist[fc].empty()) {
            freq2nodelist.erase(fc);
        }
    }
    // ++cluster_access_cnt[cid];
}

void Cache::unpin(int cid) {
    if (--cluster_pin_cnt[cid] == 0) {
        cluster_access_cnt[cid] = storage_ptr->cluster_access_cnt[cid].load();
        int fc = cluster_access_cnt[cid];
        freq2nodelist[fc].push_front(cid);
        cid2node[cid] = freq2nodelist[fc].begin();
    }
}

bool Cache::canEvit() {
    if (!free_pages.empty()) {
        return true;
    } else if (free_pages.empty() && !freq2nodelist.empty()) {
        return true;
    }
    return false;
}

idx_t Cache::evit() {
    idx_t pid = -1;
    if (!free_pages.empty()) {
        pid = free_pages.front();
        free_pages.pop();
    } else if (free_pages.empty() && !freq2nodelist.empty()) {
        auto head = freq2nodelist.begin();
        idx_t evit_cid = head->second.front();
        head->second.pop_front();
        if (head->second.empty()) {
            freq2nodelist.erase(head);
        }
        pid = cid2pid[evit_cid];
        cid2pid[evit_cid] = -1;
        pid2cid[pid] = -1;
    }
    return pid;
}

void Cache::pinEvit(int cid) {
    ++cluster_pin_cnt[cid];
}

ExGPUResources::ExGPUResources(int gpu_id, StoragePtr storage_ptr)
: device(gpu_id) {
    faiss::gpu::DeviceScope scope(gpu_id);
    top_res.getResources()->initializeForDevice(device);
    top_res.setTempMemory(faiss::hitcher::kGPUTmpMemSizeTop);
    bottom_res.getResources()->initializeForDevice(device);
    bottom_res.setTempMemory(faiss::hitcher::kGPUTmpMemSizeBottom);
    cudaStreamCreateWithFlags(&top_stream, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&bottom_stream, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&transfer_stream, cudaStreamNonBlocking);
    for (int i=0; i<64; i++) {
        cudaEvent_t event;
        cudaEventCreate(&event);
        event_pool.push(event);
    }
}

ExGPUResources::~ExGPUResources() {
    cudaStreamDestroy(top_stream);
    cudaStreamDestroy(bottom_stream);
    cudaStreamDestroy(transfer_stream);
    cudaEvent_t event;
    while (!event_pool.empty()) {
        event = event_pool.front();
        event_pool.pop();
        cudaEventDestroy(event);
    }
}

void GPUWorker::initWork(std::vector<moodycamel::ProducerToken> *token_que1, std::vector<moodycamel::ProducerToken> *token_que2, int num_token_per_thread) {
    for (int i=0; i<num_token_per_thread; i++) {
        token_que1->emplace_back(moodycamel::ProducerToken(ready_task_que_per_cls_));
        token_que2->emplace_back(moodycamel::ProducerToken(ready_task_que_per_cls_));
    }
}

GPUWorker::GPUWorker(StoragePtr storage_ptr, int k, int gpu_id, int max_query_batch_size, int max_bottom_batch_size, int max_pack_num, double top_batch_threshold, std::string kernel_mode, int transfer_batch_size)
: Worker(WorkerType::GPU, storage_ptr, k), gpu_res_(gpu_id, storage_ptr), gpu_id_(gpu_id),
  max_top_batch_size_(max_query_batch_size),
  max_bottom_batch_size_(max_bottom_batch_size), 
  max_pack_num_(max_pack_num), 
  top_batch_threshold_(top_batch_threshold),
  stop_(false),
  num_ready_batch_(0),
  num_transfering_task_(0),
  max_transfer_batch_size_(transfer_batch_size) {
    if (kernel_mode == "qc") {
        kernel_mode_ = KernelMode::QC;
    } else {
        kernel_mode_ = KernelMode::CC;
    }
    faiss::gpu::DeviceScope scope(gpu_id);
    // init cache
    gpu_cache_ = new Cache(storage_ptr, gpu_res_.top_res.getResources().get(), gpu_id, gpu_res_.transfer_stream);
    // init top searcher
    size_t indices_buffer_size = max_query_batch_size * nprobe_ * sizeof(faiss::idx_t);
    cudaMallocHost((void **)&pin_top_buffer_, indices_buffer_size);
    pin_top_indices_buffer_ = (faiss::idx_t*)pin_top_buffer_;
    // init bottom searcher
    double time_cost = omp_get_wtime();
    {
        int num_threads = 16;
        std::vector<std::vector<moodycamel::ProducerToken> > tmp_que1(num_threads);
        std::vector<std::vector<moodycamel::ProducerToken> > tmp_que2(num_threads);
        int num_token_per_thread = (storage_ptr->cluster_access_cnt.size() + num_threads - 1) / num_threads;
        std::vector<std::thread*> init_workers(num_threads);
        for (int i=0; i<num_threads; i++) {
            init_workers[i] = new std::thread(&faiss::hitcher::GPUWorker::initWork, this, &tmp_que1[i], &tmp_que2[i], num_token_per_thread);
        }
        for (int i=0; i<num_threads; i++) {
            init_workers[i]->join();
            delete init_workers[i];
            ptoks_for_ready_tasks_.insert(ptoks_for_ready_tasks_.end(), std::move_iterator(tmp_que1[i].begin()), std::move_iterator(tmp_que1[i].end()));
            ptoks_for_transfered_ready_tasks_.insert(ptoks_for_transfered_ready_tasks_.end(), std::move_iterator(tmp_que2[i].begin()), std::move_iterator(tmp_que2[i].end()));
        }
    }
    time_cost = omp_get_wtime() - time_cost;
    printf("init bottom searcher time cost: %.3f\n", time_cost);
    hitch_ride_cnt_per_cls_ready_task_.resize(storage_ptr->cluster_access_cnt.size(), 0);
    hitch_ride_cnt_per_cls_transfer_ready_task_.resize(storage_ptr->cluster_access_cnt.size(), 0);
    peek_ready_task_per_cls_.resize(storage_ptr->cluster_access_cnt.size());
    peek_transfered_ready_task_per_cls_.resize(storage_ptr->cluster_access_cnt.size());
    size_t distances_buffer_size = max_bottom_batch_size_ * k_ * sizeof(float);
    indices_buffer_size = max_bottom_batch_size_ * k_ * sizeof(faiss::idx_t);
    cudaMallocHost((void **)&pin_bottom_buffer_, distances_buffer_size + indices_buffer_size);
    pin_bottom_distances_buffer_ = (float*)pin_bottom_buffer_;
    pin_bottom_indices_buffer_ = (faiss::idx_t*)(pin_bottom_buffer_ + distances_buffer_size);
    // workers
    wait_transfer_ques_.resize(storage_ptr_->cluster_sizes.size());		    
    // batch_maker_ = new std::thread(&faiss::hitcher::GPUWorker::makeBatchWork, this); 
    // search_thread_ = new std::thread(&faiss::hitcher::GPUWorker::searchWork, this);
    // transfer_thread_ = new std::thread(&faiss::hitcher::GPUWorker::transferWork, this);
    // dispatch_thread_ = new std::thread(&faiss::hitcher::GPUWorker::dispatchWork, this);
    // launchWorker();
}

void GPUWorker::launchWorker() {
    batch_maker_ = new std::thread([this](){this->makeBatchWork();});
    search_thread_ = new std::thread([this](){this->searchWork();});
    transfer_thread_ = new std::thread([this](){this->transferWork();});
    dispatch_thread_ = new std::thread([this](){this->dispatchWork();});
}

GPUWorker::~GPUWorker() {
    stop_ = true;
    // wait for all thread to be join
    std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    if (batch_maker_ && batch_maker_->joinable()) {
        batch_maker_->join();
        delete batch_maker_;
    } else {
        printf("batch_maker_ is a nullptr\n");
    }
    if (search_thread_ && search_thread_->joinable()) {
        search_thread_->join();
        delete search_thread_;
    } else {
        printf("search_thread_ is a nullptr\n");
    }
    if (transfer_thread_ && transfer_thread_->joinable())  {
        transfer_thread_->join();
        delete transfer_thread_;
    } else {
        printf("transfer_thread_ is a nullptr\n");
    }
    if (dispatch_thread_ && dispatch_thread_->joinable()) {
        dispatch_thread_->join();
        delete dispatch_thread_;
    } else {
        printf("dispath_thread_ is a nullptr\n");
    }
    cudaFreeHost(pin_top_buffer_);
    cudaFreeHost(pin_bottom_buffer_);
    delete gpu_cache_;
}

void GPUWorker::getStats() {
    {
        int total = num_finish_task_;
        if (total) {
            printf("[GPU Worker] gpu worker finish task number: %d, num hit tasks: %d, hit rate: %.2f\n", 
                    total, num_hit_task_, (double(num_hit_task_) / total));
        }
    }

    if (make_batch_times.size()) {
        double total_time = 0;
        std::sort(make_batch_times.begin(), make_batch_times.end());
        int p50_idx = int(make_batch_times.size() * 0.5);
        int p95_idx = int(make_batch_times.size() * 0.95);
        int p99_idx = int(make_batch_times.size() * 0.99);
        double p50_time = make_batch_times[p50_idx] * 1000;
        double p95_time = make_batch_times[p95_idx] * 1000;
        double p99_time = make_batch_times[p99_idx] * 1000;
        printf("[GPUWorker] make batch time P50: %lf ms, P95: %lf ms, P99: %lf ms\n", p50_time, p95_time, p99_time);
        for (int i=0; i<make_batch_times.size(); i++) {
            total_time += make_batch_times[i];
        }
        printf("[GPUWorker] avg make batch time: %lf ms\n", (total_time / make_batch_times.size() * 1000));
    }

    if (get_ready_task_times.size()) {
        print_vec_info(get_ready_task_times, "GetReadyTask() time: ", " sec");
    }
    if (get_hitch_task_times.size()) {
        print_vec_info(get_hitch_task_times, "GetHitchTask() time: ", " sec");
    }

    {
        printf("[GPUWorker] Select Task from Ready Queue: %d, from Transfer Queue: %d\n", select_from_ready_queue, select_from_transfer_queue);
        printf("[GPUWorker] Hitch Ride Task from Ready Queue: %d, from Transfer Queue: %d\n", hitch_ride_from_ready_queue, hitch_ride_from_transfer_queue);
    }
}

void GPUWorker::addTopTask(std::vector<faiss::idx_t> &qids) {
    num_running_task_ += qids.size();
    for (auto & qid : qids) storage_ptr_->queries[qid]->top_search_gpu_id = gpu_id_;
    query_que_.enqueue_bulk(qids.begin(), qids.size());
}

void GPUWorker::addTopTask(faiss::idx_t qid) {
    num_running_task_++;
    storage_ptr_->queries[qid]->top_search_gpu_id = gpu_id_;
    query_que_.enqueue(qid);
}

void GPUWorker::addBottomTask(std::vector<Task> &tasks) {
    num_running_task_ += tasks.size();
    task_que_.enqueue_bulk(tasks.begin(), tasks.size());
}

void GPUWorker::addBottomTask(Task &task) {
    num_running_task_++;
    task_que_.enqueue(task);
}

void GPUWorker::searchBottomCC(BatchInfo &batch) {
    auto stream = gpu_res_.bottom_stream;
    int bs = batch.tasks.size();
    int dim = storage_ptr_->index->d;
    if (storage_ptr_->dtype == "int8") {
        dim /= 4;
    }
    // collect query vecs
    faiss::gpu::DeviceTensor<float, 2, true> query_vecs(
        gpu_res_.bottom_res.getResources().get(), 
        faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream), 
        {batch.unique_query_cnt, dim}
    );
    auto query_address_gpu = faiss::gpu::toDeviceTemporary<int64_t, 1>(
        gpu_res_.bottom_res.getResources().get(), 
        gpu_res_.device,
        (int64_t*)batch.unique_query_address.data(), 
        stream,
        {int(batch.unique_query_address.size())}
    );
    faiss::hitcher::gather(query_address_gpu.data(), int(batch.unique_query_address.size()), dim, query_vecs.data(), stream);
    // search process, allocate gpu result buffer
    auto query_local_ids = faiss::gpu::toDeviceTemporary<int64_t, 1>(
        gpu_res_.bottom_res.getResources().get(), 
        gpu_res_.device, 
        (int64_t*)batch.qids.data(), 
        stream,
        {bs}
    );
    auto query_offsets = faiss::gpu::toDeviceTemporary<int64_t, 1>(
        gpu_res_.bottom_res.getResources().get(), 
        gpu_res_.device,
        (int64_t*)batch.offsets.data(),
        stream,
        {int(batch.offsets.size())}
    );
    auto list_ids = faiss::gpu::toDeviceTemporary<int64_t, 1>(
        gpu_res_.bottom_res.getResources().get(), 
        gpu_res_.device,
        (int64_t*)batch.cids.data(),
        stream,
        {int(batch.cids.size())}
    );
    faiss::gpu::DeviceTensor<float, 2, true> distances(
        gpu_res_.bottom_res.getResources().get(), 
        faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream), 
        {bs, k_}
    );
    faiss::gpu::DeviceTensor<idx_t, 2, true> indices(
        gpu_res_.bottom_res.getResources().get(), 
        faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream), 
        {bs, k_}
    );
    faiss::gpu::runIVFFlatScanHCC(
        query_vecs,
        query_local_ids,
        query_offsets,
        list_ids,
        gpu_cache_->d_cluster_codes_ptrs,
        gpu_cache_->d_cluster_ids_ptrs,
        gpu_cache_->d_cluster_sizes,
        kNumVecPerCluster,
        k_,
        storage_ptr_->index->metric_type,
        distances,
        indices,
        gpu_res_.bottom_res.getResources().get(),
        gpu_res_.bottom_stream
    );
    // copy result back to cpu
    bs = std::min(bs, max_bottom_batch_size_);
    cudaMemcpyAsync(pin_bottom_distances_buffer_, distances.data(), bs * k_ * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(pin_bottom_indices_buffer_, indices.data(), bs * k_ * sizeof(idx_t), cudaMemcpyDeviceToHost, stream);
    // BatchPartialResult batch_partial_result;
    // batch_partial_result.tasks = batch.tasks;
    // cudaStreamSynchronize(stream);
    // batch_partial_result.distances = std::vector<float>(pin_bottom_distances_buffer_, pin_bottom_distances_buffer_ + k_ * bs);
    // batch_partial_result.result_indices = std::vector<idx_t>(pin_bottom_indices_buffer_, pin_bottom_indices_buffer_ + k_ * bs);
    // storage_ptr_->addBatchPartialResult(batch_partial_result);
}

void GPUWorker::searchBottomQC(BatchInfo &batch) {
    auto stream = gpu_res_.bottom_stream;
    int num_queries = batch.qids.size();
    int dim = storage_ptr_->index->d;
    if (storage_ptr_->dtype == "int8") {
        dim /= 4;
    }
    // collect query vecs
    faiss::gpu::DeviceTensor<float, 2, true> query_vecs(
        gpu_res_.bottom_res.getResources().get(), 
        faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream), 
        {num_queries, dim}
    );
    auto query_address_gpu = faiss::gpu::toDeviceTemporary<int64_t, 1>(
        gpu_res_.bottom_res.getResources().get(), 
        gpu_res_.device,
        (int64_t*)batch.unique_query_address.data(), 
        stream,
        {num_queries}
    );
    faiss::hitcher::gather(query_address_gpu.data(), num_queries, dim, query_vecs.data(), stream);
    // search process, allocate gpu result buffer
    int max_probe = batch.cids.size() / num_queries;
    auto coarse_indices = faiss::gpu::toDeviceTemporary<int64_t, 2>(
        gpu_res_.bottom_res.getResources().get(), 
        gpu_res_.device,
        (int64_t*)batch.cids.data(),
        stream,
        {num_queries, max_probe}
    );
    faiss::gpu::DeviceTensor<float, 2, true> distances(
        gpu_res_.bottom_res.getResources().get(), 
        faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream), 
        {num_queries, k_}
    );
    faiss::gpu::DeviceTensor<idx_t, 2, true> indices(
        gpu_res_.bottom_res.getResources().get(), 
        faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream), 
        {num_queries, k_}
    );
    faiss::gpu::runIVFFlatScanHQC(
        query_vecs,
        coarse_indices,
        num_queries,
        gpu_cache_->d_cluster_codes_ptrs,
        gpu_cache_->d_cluster_ids_ptrs,
        gpu_cache_->d_cluster_sizes,
        kNumVecPerCluster,
        k_,
        storage_ptr_->index->metric_type,
        distances,
        indices,
        gpu_res_.bottom_res.getResources().get(),
        gpu_res_.bottom_stream
    );
    // copy result back to cpu
    int bs = std::min(int(batch.tasks.size()), int(max_bottom_batch_size_));
    cudaMemcpyAsync(pin_bottom_distances_buffer_, distances.data(), bs * k_ * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(pin_bottom_indices_buffer_, indices.data(), bs * k_ * sizeof(idx_t), cudaMemcpyDeviceToHost, stream);
    // BatchPartialResult batch_partial_result;
    // batch_partial_result.tasks = batch.tasks;
    // cudaStreamSynchronize(stream);
    // batch_partial_result.distances = std::vector<float>(pin_bottom_distances_buffer_, pin_bottom_distances_buffer_ + k_ * bs);
    // batch_partial_result.result_indices = std::vector<idx_t>(pin_bottom_indices_buffer_, pin_bottom_indices_buffer_ + k_ * bs);
    // storage_ptr_->addBatchPartialResult(batch_partial_result);
}

void GPUWorker::syncSearchBottom(BatchInfo &batch) {
    auto stream = gpu_res_.bottom_stream;
    cudaStreamSynchronize(stream);
    finish_search_task_que_.enqueue_bulk(batch.tasks.begin(), batch.tasks.size());
}

bool GPUWorker::getReadyTask(Task& task) {
    auto ts = omp_get_wtime();
    if (!peek_ready_task_.has_value()) {
        Task tmp;
        if (ready_tasks_.try_dequeue(tmp)) {
            peek_ready_task_ = tmp;
        }
    }

    if (!peek_transfered_ready_task_.has_value()) {
        Task tmp;
        if (transfered_ready_tasks_.try_dequeue(tmp)) {
            peek_transfered_ready_task_ = tmp;
        }
    }

    if (peek_ready_task_.has_value() && peek_transfered_ready_task_.has_value()) {
        if (peek_ready_task_->query_id < peek_transfered_ready_task_->query_id) {
            task = peek_ready_task_.value();
            peek_ready_task_.reset();
            select_from_ready_queue++;
        } else {
            task = peek_transfered_ready_task_.value();
            peek_transfered_ready_task_.reset();
            select_from_transfer_queue++;
        }
        auto dur = omp_get_wtime() - ts;
        get_ready_task_times.emplace_back(dur);
        return true;
    }

    if (peek_ready_task_.has_value()) {
        task = peek_ready_task_.value();
        peek_ready_task_.reset();
        select_from_ready_queue++;
        auto dur = omp_get_wtime() - ts;
        get_ready_task_times.emplace_back(dur);
        return true;
    }

    if (peek_transfered_ready_task_.has_value()) {
        task = peek_transfered_ready_task_.value();
        peek_transfered_ready_task_.reset();
        select_from_transfer_queue++;
        auto dur = omp_get_wtime() - ts;
        get_ready_task_times.emplace_back(dur);
        return true;
    }

    return false;
}

void GPUWorker::removeTaskFromClsQues(const Task task) {
    if (task.wait_transfer) {
        if (peek_transfered_ready_task_per_cls_[task.cluster_id].has_value()) {
            if (peek_transfered_ready_task_per_cls_[task.cluster_id].value() == task) {
                peek_transfered_ready_task_per_cls_[task.cluster_id].reset();
            } else {
                assert(false && "no task in cls que after selection, peek != select");
            }
        } else {
            Task tmp;
            if (transfered_ready_task_que_per_cls_.try_dequeue_from_producer(ptoks_for_transfered_ready_tasks_[task.cluster_id], tmp)) {
                assert(tmp == task);
            } else {
                assert(false && "no task in cls que after selection, head != select");
            }
        }
    } else {
        if (peek_ready_task_per_cls_[task.cluster_id].has_value()) {
            if (peek_ready_task_per_cls_[task.cluster_id].value() == task) {
                peek_ready_task_per_cls_[task.cluster_id].reset();
            } else {
                assert(false && "no task in cls que after selection, peek != select");
            }
        } else {
            Task tmp;
            if (ready_task_que_per_cls_.try_dequeue_from_producer(ptoks_for_ready_tasks_[task.cluster_id], tmp)) {
                assert(tmp == task);
            } else {
                assert(false && "no task in cls que after selection, head != select");
            }
        }
    }
}

bool GPUWorker::getHitchRideTask(Task& task, idx_t cluster_id) {
    // ensure the tasks are firstly selected from the same source?
    auto ts = omp_get_wtime();
    if (!peek_transfered_ready_task_per_cls_[cluster_id].has_value()) {
        Task tmp;
        if (transfered_ready_task_que_per_cls_.try_dequeue_from_producer(ptoks_for_transfered_ready_tasks_[cluster_id], tmp)) {
            peek_transfered_ready_task_per_cls_[cluster_id] = tmp;
        }
    }
    if (!peek_ready_task_per_cls_[cluster_id].has_value()) {
        Task tmp;
        if (ready_task_que_per_cls_.try_dequeue_from_producer(ptoks_for_ready_tasks_[cluster_id], tmp)) {
            peek_ready_task_per_cls_[cluster_id] = tmp;
        }
    }

    if (peek_ready_task_per_cls_[cluster_id].has_value() && 
        peek_transfered_ready_task_per_cls_[cluster_id].has_value()) {
        if (peek_ready_task_per_cls_[cluster_id]->query_id < peek_transfered_ready_task_per_cls_[cluster_id]->query_id) {
            task = peek_ready_task_per_cls_[cluster_id].value();
            peek_ready_task_per_cls_[cluster_id].reset();
            hitch_ride_from_ready_queue++;
        } else {
            task = peek_transfered_ready_task_per_cls_[cluster_id].value();
            peek_transfered_ready_task_per_cls_[cluster_id].reset();
            hitch_ride_from_transfer_queue++;
        }
        auto dur = omp_get_wtime() - ts;
        get_hitch_task_times.emplace_back(dur);
        return true;
    }

    if (peek_ready_task_per_cls_[cluster_id].has_value()) {
        task = peek_ready_task_per_cls_[cluster_id].value();
        peek_ready_task_per_cls_[cluster_id].reset();
        hitch_ride_from_ready_queue++;
        auto dur = omp_get_wtime() - ts;
        get_hitch_task_times.emplace_back(dur);
        return true;
    }

    if (peek_transfered_ready_task_per_cls_[cluster_id].has_value()) {
        task = peek_transfered_ready_task_per_cls_[cluster_id].value();
        peek_transfered_ready_task_per_cls_[cluster_id].reset();
        hitch_ride_from_transfer_queue++;
        auto dur = omp_get_wtime() - ts;
        get_hitch_task_times.emplace_back(dur);
        return true;
    }

    return false;
}

void GPUWorker::selectCC(BatchInfo &batch) {
    Task task;
    std::unordered_map<faiss::idx_t, std::vector<idx_t> > cid2qids;
    std::unordered_map<idx_t, idx_t> unique_queries;
    int dim = storage_ptr_->index->d;
    batch.unique_query_cnt = 0;
    batch.num_queries = 0;
    batch.num_cls = 0;
    int hitch_ride_cnt = 0;
    while (getReadyTask(task)) {
    // while (ready_tasks_.try_dequeue(task)) {
        // done in hitch ride
        if (task.wait_transfer) {
            if (hitch_ride_cnt_per_cls_transfer_ready_task_[task.cluster_id] > 0) {
                hitch_ride_cnt_per_cls_transfer_ready_task_[task.cluster_id]--;
                continue;
            };
        } else {
            if (hitch_ride_cnt_per_cls_ready_task_[task.cluster_id] > 0) {
                hitch_ride_cnt_per_cls_ready_task_[task.cluster_id]--;
                continue;
            };
        }

        Task selected = task;
        idx_t cur_cluster_id = task.cluster_id;

        batch.tasks.emplace_back(task);
        ++batch.num_queries;
        if (unique_queries.find(task.query_id) == unique_queries.end()) {
            unique_queries[task.query_id] = batch.unique_query_cnt++;
            batch.unique_query_address.emplace_back(storage_ptr_->query_vecs_float32 + task.query_id * dim);
        }
        cid2qids[task.cluster_id].emplace_back(unique_queries[task.query_id]);
        // get ride of the selected task in queue
        removeTaskFromClsQues(selected);

        // hitch ride
        for (int i=1; i<max_pack_num_; i++) {
            if (getHitchRideTask(task, cur_cluster_id)) {
            // if (ready_task_que_per_cls_.try_dequeue_from_producer(ptoks_[task.cluster_id], task)) {
                batch.tasks.emplace_back(task);
                if (unique_queries.find(task.query_id) == unique_queries.end()) {
                    unique_queries[task.query_id] = batch.unique_query_cnt++;
                    batch.unique_query_address.emplace_back(storage_ptr_->query_vecs_float32 + task.query_id * dim);
                }
                // cid2qids[task.cluster_id].emplace_back(task.query_id);
                cid2qids[task.cluster_id].emplace_back(unique_queries[task.query_id]);
                ++batch.num_queries;
                hitch_ride_cnt++;
                if (task.wait_transfer) {
                    hitch_ride_cnt_per_cls_transfer_ready_task_[cur_cluster_id]++;
                } else {
                    hitch_ride_cnt_per_cls_ready_task_[cur_cluster_id]++;
                }
            } else {
                break;
            }
        }
        if (++batch.num_cls >= max_bottom_batch_size_) {
            break;
        }
    }
    if (batch.num_queries == 0) {
        return;
    }
    idx_t prefix = 0;
    batch.offsets.emplace_back(0);
    batch.qids.reserve(batch.num_queries);
    for (auto &iter : cid2qids) {
        std::vector<idx_t> &qids = iter.second;
        int num_seg = (qids.size() + max_pack_num_ - 1) / max_pack_num_;
        idx_t remap_cid = gpu_cache_->cid2pid[iter.first];
        for (int i=0; i<num_seg; i++) {
            batch.cids.emplace_back(remap_cid);
            int seg_size = std::min(max_pack_num_, int(qids.size()) - i * max_pack_num_);
            prefix += seg_size;
            batch.offsets.emplace_back(prefix);
        }
        batch.qids.insert(batch.qids.end(), std::move_iterator(qids.begin()), std::move_iterator(qids.end()));
    }

    {
        // gather batch stats for analysis
        TaskBatchStats stat;
        stat.num_cls = batch.num_cls;
        stat.num_unique_cls = batch.num_cls;
        stat.num_queries = hitch_ride_cnt;
        stat.num_unique_queries = batch.unique_query_cnt;
        storage_ptr_->task_batch_stats_lists.enqueue(std::move(stat));
    }
    // printf("[GPUWorker][CC] Number of clusters: %d, Number of queries: %d, Number of unique queries: %d, Number of hitch ride: %d\n",
    //     batch.num_cls, batch.num_queries, batch.unique_query_cnt, hitch_ride_cnt);
}

void GPUWorker::selectQC(BatchInfo &batch) {
    Task task;
    std::unordered_map<faiss::idx_t, std::vector<idx_t> > qid2cids;
    batch.num_queries = 0;
    int max_probe = 0;
    while (getReadyTask(task)) {
        // ready_task_que_per_cls_.try_dequeue_from_producer(ptoks_[task.cluster_id], task);
        batch.tasks.push_back(task);
        idx_t remap_cid = gpu_cache_->cid2pid[task.cluster_id];
        qid2cids[task.query_id].push_back(remap_cid);
        if (qid2cids[task.query_id].size() > max_probe) {
            max_probe = qid2cids[task.query_id].size();
        }
        if (++batch.num_queries >= max_bottom_batch_size_) {
            break;
        }
    }
    if (batch.num_queries == 0) {
        return;
    }
    int dim = storage_ptr_->index->d;
    batch.unique_query_cnt = 0;
    int valid_cls_cnt = 0;
    for (auto &iter : qid2cids) {
        auto qid = iter.first;
        batch.qids.push_back(qid);
        batch.unique_query_address.push_back(storage_ptr_->query_vecs_float32 + qid * dim);
        batch.unique_query_cnt++;
        std::vector<idx_t> &cids = iter.second;
        int num_cids = cids.size();
        valid_cls_cnt += num_cids;
        batch.cids.insert(batch.cids.end(), std::move_iterator(cids.begin()), std::move_iterator(cids.end()));
        while (num_cids++ < max_probe) {
            batch.cids.push_back(-1); //padding
        }
    }

    {
        // gather batch stats for analysis
        TaskBatchStats stat;
        stat.num_cls = valid_cls_cnt;
        stat.num_unique_cls = valid_cls_cnt;
        stat.num_queries = batch.qids.size();
        stat.num_unique_queries = batch.unique_query_cnt;
        storage_ptr_->task_batch_stats_lists.enqueue(std::move(stat));
    }
    // printf("[GPUWorker][QC] Number of tasks: %d, Number of unique queries: %d\n", batch.num_queries, batch.unique_query_cnt);
}

void GPUWorker::makeBatchWork() {
    while (!stop_) {
        if (num_ready_batch_ > 0) {
            continue;
        }
        double tc = omp_get_wtime();
        BatchInfo batch;
        if (kernel_mode_ == KernelMode::CC) {
            selectCC(batch);
        } else {
            selectQC(batch);
        }
        tc = omp_get_wtime() - tc;
        if (batch.num_queries > 0) {
            ready_batch_.enqueue(batch);
            num_ready_batch_++;
            make_batch_times.push_back(tc);
        }
    }
}

void GPUWorker::searchTop(std::vector<idx_t> &query_batch) {
    auto stream = gpu_res_.top_stream;
    int bs = query_batch.size();
    // printf("[GPUWorker][searchTop] batch size: %d\n", bs);
    int dim = storage_ptr_->index->d;
    // collect query vecs
    std::vector<float*> query_vec_address(bs);
    double ts = omp_get_wtime();
    for (int i=0; i<query_batch.size(); i++) {
        query_vec_address[i] = storage_ptr_->query_vecs_float32 + (query_batch[i] * dim);
        storage_ptr_->queries[query_batch[i]]->top_execution_start_time = ts;
    }
    auto query_vec_address_gpu = faiss::gpu::toDeviceTemporary<int64_t, 1>(
        gpu_res_.top_res.getResources().get(), 
        gpu_res_.device, 
        (int64_t*)query_vec_address.data(), 
        stream,
        {bs}
    );
    faiss::gpu::DeviceTensor<float, 2, true> query_vecs(
        gpu_res_.top_res.getResources().get(), 
        faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream), 
        {bs, dim}
    );
    faiss::hitcher::gather(query_vec_address_gpu.data(), bs, dim, query_vecs.data(), stream);
    // search process, allocate gpu result buffer
    faiss::gpu::DeviceTensor<float, 2, true> distances(
        gpu_res_.top_res.getResources().get(), 
        faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream), 
        {bs, nprobe_}
    );
    faiss::gpu::DeviceTensor<idx_t, 2, true> indices(
        gpu_res_.top_res.getResources().get(), 
        faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream), 
        {bs, nprobe_}
    );
    faiss::gpu::bfKnnOnDevice(
        gpu_res_.top_res.getResources().get(),
        gpu_res_.device,
        stream,
        gpu_cache_->d_centroids,
        true,
        &gpu_cache_->d_centroids_norms,
        query_vecs,
        true,
        nprobe_,
        storage_ptr_->index->metric_type,
        0,
        distances,
        indices,
        true
    );
    // copy result back to cpu
    cudaMemcpyAsync(pin_top_indices_buffer_, indices.data(), bs * nprobe_ * sizeof(idx_t), cudaMemcpyDeviceToHost, stream);
    // cudaStreamSynchronize(stream);
    // double finish_time = omp_get_wtime();
    // for (int idx = 0; idx < bs; idx++) {
    //     idx_t qid = query_batch[idx];
    //     idx_t *data_ptr = pin_top_indices_buffer_ + idx * nprobe_;
    //     storage_ptr_->queries[qid]->cids = std::vector<idx_t>(data_ptr, data_ptr + nprobe_);
    //     storage_ptr_->queries[qid]->finish_top_time = finish_time;
    // }
    // storage_ptr_->finish_top_queries.enqueue_bulk(query_batch.begin(), query_batch.size());
}

void GPUWorker::syncSearchTop(std::vector<idx_t> &query_batch) {
    auto stream = gpu_res_.top_stream;
    int bs = query_batch.size();
    cudaStreamSynchronize(stream);
    double finish_time = omp_get_wtime();
    for (int idx = 0; idx < bs; idx++) {
        idx_t qid = query_batch[idx];
        idx_t *data_ptr = pin_top_indices_buffer_ + idx * nprobe_;
        storage_ptr_->queries[qid]->cids = std::vector<idx_t>(data_ptr, data_ptr + nprobe_);
        storage_ptr_->queries[qid]->finish_top_time = finish_time;
    }
    storage_ptr_->finish_top_queries.enqueue_bulk(query_batch.begin(), query_batch.size());
}

void GPUWorker::searchWork() {
    faiss::gpu::DeviceScope scope(gpu_res_.device);
    while (!stop_) {
        // launch search top
        faiss::idx_t qid;
        std::vector<faiss::idx_t> query_batch;
        double st = omp_get_wtime();
        while (true) {
            if (query_que_.try_dequeue(qid)) {
                query_batch.push_back(qid);
                if (query_batch.size() >= max_top_batch_size_) {
                    break;
                }
            }
            double dur = omp_get_wtime() - st;
            if (dur > top_batch_threshold_) {
                break;
            }
        }
        if (query_batch.size() > 0) {
            searchTop(query_batch);
        }

        // launch search bottom
        BatchInfo batch;
        if (ready_batch_.try_dequeue(batch)) {
            num_ready_batch_--;
            if (kernel_mode_ == KernelMode::CC) {
                searchBottomCC(batch);
            } else {
                searchBottomQC(batch);
            }
        }
        // sync search top
        if (query_batch.size() > 0) {
            syncSearchTop(query_batch);
            num_running_task_ -= query_batch.size();
            query_batch.clear();
        }
        // sync search bottom
        if (batch.tasks.size() > 0) {
            syncSearchBottom(batch);
            num_finish_task_ += batch.tasks.size();
            num_running_task_ -= batch.tasks.size();
        }
    }
}

void GPUWorker::transfer(Task &task, cudaEvent_t event) {
    auto stream = gpu_res_.transfer_stream;
    idx_t cid = task.cluster_id;
    idx_t pid = gpu_cache_->cid2pid[cid];
    idx_t c_size = storage_ptr_->getClusterSize(cid);
    int8_t *c_data = storage_ptr_->getClusterCodes(cid);
    cudaMemcpyAsync(
        gpu_cache_->d_cluster_data[pid],
        c_data,
        c_size * storage_ptr_->ele_bytes * storage_ptr_->dim,
        cudaMemcpyHostToDevice,
        stream
    );
    gpu_cache_->d_cluster_sizes.setAt(pid, c_size, stream);
    cudaEventRecord(event, stream);
}

void GPUWorker::transferWork() {
    // printf("gpu worker transfer work\n");
    faiss::gpu::DeviceScope scope(gpu_res_.device);
    while (!stop_) {
        Task task;
        // wait for complete
        while (!transfering_que_.empty()) {
            auto &event = transfer_event_queue_.front();
            if (cudaEventQuery(event) == cudaSuccess) {
                transfer_event_queue_.pop();
                task = transfering_que_.front();
                transfering_que_.pop();
                finish_transfer_task_que_.enqueue(task);
                gpu_res_.event_pool.push(event);
                num_transfering_task_--;
            } else {
                break;
            }
        }
        // transfer new task
        while (!pending_que_.empty() && !gpu_res_.event_pool.empty()) {
            task = pending_que_.front();
            pending_que_.pop();
            cudaEvent_t event = gpu_res_.event_pool.front();
            gpu_res_.event_pool.pop();
            transfer(task, event);
            transfering_que_.push(task);
            transfer_event_queue_.push(event);
        }
        while (transfer_task_que_.try_dequeue(task)) {
            if (!gpu_res_.event_pool.empty()) {
                cudaEvent_t event = gpu_res_.event_pool.front();
                gpu_res_.event_pool.pop();
                transfer(task, event);
                transfering_que_.push(task);
                transfer_event_queue_.push(event);
            } else {
                pending_que_.push(task);
                break;
            }
        }
    }
}

void GPUWorker::addReadyTask(Task &task, bool after_transfered) {
    if (after_transfered) {
        task.set_wait_transfer();
        transfered_ready_task_que_per_cls_.enqueue(ptoks_for_transfered_ready_tasks_[task.cluster_id], task);
        transfered_ready_tasks_.enqueue(task);
    } else {
        ready_task_que_per_cls_.enqueue(ptoks_for_ready_tasks_[task.cluster_id], task);
        ready_tasks_.enqueue(task);
    }
}

void GPUWorker::addReadyTask(std::vector<Task> &tasks, bool after_transfered) {
    if (tasks.size() == 0) return;
    if (after_transfered) {
        for (auto& t : tasks) {
            t.set_wait_transfer();
            transfered_ready_task_que_per_cls_.enqueue(ptoks_for_transfered_ready_tasks_[t.cluster_id], t);
        }
        transfered_ready_tasks_.enqueue_bulk(tasks.begin(), tasks.size());
    } else {
        for (auto &t : tasks) {
            ready_task_que_per_cls_.enqueue(ptoks_for_ready_tasks_[t.cluster_id], t);
        }
        ready_tasks_.enqueue_bulk(tasks.begin(), tasks.size());
    }
}

bool GPUWorker::stealTaskFrom(Task &task) {
    if (wait_page_task_que_.try_dequeue(task)) {
        num_running_task_--;
        return true;
    }
    if (miss_task_que_.try_dequeue(task)) {
        num_running_task_--;
        return true;
    }
    return false;
}

void GPUWorker::dispatchWork() {
    while (!stop_) {
        Task task;
        std::vector<Task> ready_task_buffer;
        std::vector<Task> transfered_ready_task_buffer;
        // unpin finish tasks
        while (finish_search_task_que_.try_dequeue(task)) {
            gpu_cache_->unpin(task.cluster_id);
            storage_ptr_->finish_tasks.enqueue(task);
        }
        // deal with finish transfer task
        while (finish_transfer_task_que_.try_dequeue(task)) {
            while (!wait_transfer_ques_[task.cluster_id].empty()) {
                // addReadyTask(wait_transfer_ques_[task.cluster_id].front(), false);
                transfered_ready_task_buffer.emplace_back(wait_transfer_ques_[task.cluster_id].front());
                wait_transfer_ques_[task.cluster_id].pop();
            }
        }
        addReadyTask(transfered_ready_task_buffer, true);
        // classify new tasks
        while (task_que_.try_dequeue(task)) {
            storage_ptr_->cluster_access_cnt[task.cluster_id]++;
            if (!wait_transfer_ques_[task.cluster_id].empty()) {
                gpu_cache_->pin(task.cluster_id);
                wait_transfer_ques_[task.cluster_id].push(task);
                num_hit_task_++;
            } else if (gpu_cache_->hit(task.cluster_id)) {
                gpu_cache_->pin(task.cluster_id);
                // addReadyTask(task);
                ready_task_buffer.emplace_back(task);
                num_hit_task_++;
            } else {
                miss_task_que_.enqueue(task);
            }
        }
        addReadyTask(ready_task_buffer, false);
        // deal with miss
        while (num_transfering_task_ < max_transfer_batch_size_) {
            if (wait_page_task_que_.try_dequeue(task)) {
                if (!wait_transfer_ques_[task.cluster_id].empty()) {
                    gpu_cache_->pin(task.cluster_id);
                    wait_transfer_ques_[task.cluster_id].push(task);
                } else if (gpu_cache_->hit(task.cluster_id)) {
                    gpu_cache_->pin(task.cluster_id);
                    addReadyTask(task, true);
                    num_hit_task_++;
                } else {
                    idx_t pid = gpu_cache_->evit();
                    if (pid >= 0) {
                        gpu_cache_->cid2pid[task.cluster_id] = pid;
                        gpu_cache_->pid2cid[pid] = task.cluster_id;
                        gpu_cache_->pinEvit(task.cluster_id);
                        transfer_task_que_.enqueue(task);
                        wait_transfer_ques_[task.cluster_id].push(task);
                        num_transfering_task_++;
                    } else {
                        wait_page_task_que_.enqueue(task);
                        break;
                    }
                }
            } else if (miss_task_que_.try_dequeue(task)) {
                if (!wait_transfer_ques_[task.cluster_id].empty()) {
                    gpu_cache_->pin(task.cluster_id);
                    wait_transfer_ques_[task.cluster_id].push(task);
                } else if (gpu_cache_->hit(task.cluster_id)) {
                    gpu_cache_->pin(task.cluster_id);
                    addReadyTask(task, true);
                    num_hit_task_++;
                } else {
                    idx_t pid = gpu_cache_->evit();
                    if (pid >= 0) {
                        gpu_cache_->cid2pid[task.cluster_id] = pid;
                        gpu_cache_->pid2cid[pid] = task.cluster_id;
                        gpu_cache_->pinEvit(task.cluster_id);
                        transfer_task_que_.enqueue(task);
                        wait_transfer_ques_[task.cluster_id].push(task);
                        num_transfering_task_++;
                    } else {
                        wait_page_task_que_.enqueue(task);
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }
}

RummyGPUWorker::RummyGPUWorker(StoragePtr storage_ptr, int k, int gpu_id, int max_query_batch_size, int max_bottom_batch_size, int max_pack_num, double top_batch_threshold, std::string kernel_mode, int transfer_batch_size)
: GPUWorker(storage_ptr, k, gpu_id, max_query_batch_size, max_bottom_batch_size, max_pack_num, top_batch_threshold, kernel_mode, transfer_batch_size) {
  num_batch_task_to_dispatch_ = 0;
  num_batch_task_remain_ = 0;
}

// void RummyGPUWorker::launchWorker() {
//     batch_maker_ = new std::thread([this](){this->makeBatchWork();});
//     search_thread_ = new std::thread([this](){this->searchWork();});
//     transfer_thread_ = new std::thread([this](){this->transferWork();});
//     dispatch_thread_ = new std::thread([this](){this->dispatchWork();});
// }

RummyGPUWorker::~RummyGPUWorker() {}

bool RummyGPUWorker::stealTaskFrom(Task &task) {
  return false;
}

void RummyGPUWorker::syncSearchTop2(std::vector<idx_t> &query_batch) {
  auto stream = gpu_res_.top_stream;
  int bs = query_batch.size();
  cudaStreamSynchronize(stream);
  double finish_time = omp_get_wtime();
  for (int idx = 0; idx < bs; idx++) {
    idx_t qid = query_batch[idx];
    idx_t *data_ptr = pin_top_indices_buffer_ + idx * nprobe_;
    storage_ptr_->queries[qid]->cids = std::vector<idx_t>(data_ptr, data_ptr + nprobe_);
    storage_ptr_->queries[qid]->finish_top_time = finish_time;
    for (auto cid : storage_ptr_->queries[qid]->cids) {
      num_batch_task_to_dispatch_ += storage_ptr_->id_map[cid].size();
      num_batch_task_remain_ += storage_ptr_->id_map[cid].size();
    }
    storage_ptr_->finish_top_queries.enqueue(qid);
  }
  // printf("num task to dispatch: %d\n", num_batch_task_to_dispatch_.load());
  // storage_ptr_->finish_top_queries.enqueue_bulk(query_batch.begin(), query_batch.size());
}

void RummyGPUWorker::selectBatch(BatchInfo &batch) {
    Task task;
    std::unordered_map<faiss::idx_t, std::vector<idx_t> > qid2cids;
    batch.num_queries = 0;
    int max_probe = 0;
    idx_t cur_cid;
    while (ready_cids_.try_dequeue(cur_cid)) {
        while (ready_task_que_per_cls_.try_dequeue_from_producer(ptoks_for_ready_tasks_[cur_cid], task)) {
            batch.tasks.emplace_back(task);
            idx_t remap_cid = gpu_cache_->cid2pid[task.cluster_id];
            qid2cids[task.query_id].push_back(remap_cid);
            if (qid2cids[task.query_id].size() > max_probe) {
                max_probe = qid2cids[task.query_id].size();
            }
            ++batch.num_queries;
        }
        if (batch.num_queries >= max_bottom_batch_size_) {
            break;
        }
    }
    if (batch.num_queries == 0) {
        return;
    }
    int dim = storage_ptr_->index->d;
    batch.unique_query_cnt = 0;
    int valid_cls_cnt = 0;
    for (auto &iter : qid2cids) {
        auto qid = iter.first;
        batch.qids.push_back(qid);
        batch.unique_query_address.push_back(storage_ptr_->query_vecs_float32 + qid * dim);
        batch.unique_query_cnt++;
        std::vector<idx_t> &cids = iter.second;
        int num_cids = cids.size();
        valid_cls_cnt += num_cids;
        batch.cids.insert(batch.cids.end(), std::move_iterator(cids.begin()), std::move_iterator(cids.end()));
        while (num_cids++ < max_probe) {
            batch.cids.push_back(-1); //padding
        }
    }

    {
        // gather batch stats for analysis
        TaskBatchStats stat;
        stat.num_cls = valid_cls_cnt;
        stat.num_unique_cls = valid_cls_cnt;
        stat.num_queries = batch.qids.size();
        stat.num_unique_queries = batch.unique_query_cnt;
        storage_ptr_->task_batch_stats_lists.enqueue(std::move(stat));
    }
}

void RummyGPUWorker::makeBatchWork() {
    while (!stop_) {
        if (num_ready_batch_ > 0) {
            continue;
        }
        BatchInfo batch;
        selectBatch(batch);
        if (batch.num_queries > 0) {
            ready_batch_.enqueue(batch);
            num_ready_batch_++;
        }
    }
}

void RummyGPUWorker::searchWork() {
//   printf("rummy gpu worker search work\n");
  num_batch_task_to_dispatch_ = 0;
  num_batch_task_remain_ = 0;
  faiss::gpu::DeviceScope scope(gpu_res_.device);
  while (!stop_) {
    std::vector<faiss::idx_t> query_batch;
    faiss::idx_t qid;
    auto st = omp_get_wtime();
    while (true) {
        if (query_que_.try_dequeue(qid)) {
            query_batch.push_back(qid);
            if (query_batch.size() >= max_top_batch_size_) {
                break;
            }
        }
        double dur = omp_get_wtime() - st;
        if (dur > top_batch_threshold_) {
            break;
        }
    }

    if (query_batch.size() > 0) {
        searchTop(query_batch);
        syncSearchTop2(query_batch);
        storage_ptr_->count_unique_access_cluster_rate(query_batch);
    }

    // sync dispatch
    while (num_batch_task_to_dispatch_ > 0);
    while (num_batch_task_remain_ > 0) {
      BatchInfo batch;
      if (ready_batch_.try_dequeue(batch)) {
        num_ready_batch_--;
        searchBottomQC(batch);
        syncSearchBottom(batch);
        num_finish_task_ += batch.tasks.size();
        num_batch_task_remain_ -= batch.tasks.size();
      }
    }
    storage_ptr_->batch_finish_ = true;
  }
}

void RummyGPUWorker::dispatchWork() {
    // printf("rummy gpu worker dispatch work\n");
    while (!stop_) {
        Task task;
        // unpin finish tasks
        while (finish_search_task_que_.try_dequeue(task)) {
            gpu_cache_->unpin(task.cluster_id);
            storage_ptr_->finish_tasks.enqueue(task);
        }
        // deal with finish transfer task
        while (finish_transfer_task_que_.try_dequeue(task)) {
            while (!wait_transfer_ques_[task.cluster_id].empty()) {
                task = wait_transfer_ques_[task.cluster_id].front();
                ready_task_que_per_cls_.enqueue(ptoks_for_ready_tasks_[task.cluster_id], task);
                wait_transfer_ques_[task.cluster_id].pop();
            }
            ready_cids_.enqueue(task.cluster_id);
        }
        // classify new tasks
        while (task_que_.try_dequeue(task)) {
            storage_ptr_->cluster_access_cnt[task.cluster_id]++;
            if (!wait_transfer_ques_[task.cluster_id].empty()) {
                gpu_cache_->pin(task.cluster_id);
                wait_transfer_ques_[task.cluster_id].push(task);
                num_hit_task_++;
            } else if (gpu_cache_->hit(task.cluster_id)) {
                gpu_cache_->pin(task.cluster_id);
                ready_task_que_per_cls_.enqueue(ptoks_for_ready_tasks_[task.cluster_id], task);
                ready_cids_.enqueue(task.cluster_id);
                num_hit_task_++;
            } else {
                miss_task_que_.enqueue(task);
            }
            num_batch_task_to_dispatch_--;
            // printf("[GPU %d] num_batch_task_to_dispatch_: %d\n", gpu_id_, num_batch_task_to_dispatch_.load());
        }
        // deal with miss
        while (wait_page_task_que_.try_dequeue(task)) {
            if (!wait_transfer_ques_[task.cluster_id].empty()) {
                gpu_cache_->pin(task.cluster_id);
                wait_transfer_ques_[task.cluster_id].push(task);
            } else if (gpu_cache_->hit(task.cluster_id)) {
                gpu_cache_->pin(task.cluster_id);
                ready_task_que_per_cls_.enqueue(ptoks_for_ready_tasks_[task.cluster_id], task);
                ready_cids_.enqueue(task.cluster_id);
                num_hit_task_++;
            } else {
                idx_t pid = gpu_cache_->evit();
                if (pid >= 0) {
                    gpu_cache_->cid2pid[task.cluster_id] = pid;
                    gpu_cache_->pid2cid[pid] = task.cluster_id;
                    gpu_cache_->pinEvit(task.cluster_id);
                    transfer_task_que_.enqueue(task);
                    wait_transfer_ques_[task.cluster_id].push(task);
                    num_transfering_task_++;
                } else {
                    wait_page_task_que_.enqueue(task);
                    break;
                }
            }
        }

        while (miss_task_que_.try_dequeue(task)) {
            if (!wait_transfer_ques_[task.cluster_id].empty()) {
                gpu_cache_->pin(task.cluster_id);
                wait_transfer_ques_[task.cluster_id].push(task);
            } else if (gpu_cache_->hit(task.cluster_id)) {
                gpu_cache_->pin(task.cluster_id);
                ready_task_que_per_cls_.enqueue(ptoks_for_ready_tasks_[task.cluster_id], task);
                ready_cids_.enqueue(task.cluster_id);
                num_hit_task_++;
            } else {
                idx_t pid = gpu_cache_->evit();
                if (pid >= 0) {
                    gpu_cache_->cid2pid[task.cluster_id] = pid;
                    gpu_cache_->pid2cid[pid] = task.cluster_id;
                    gpu_cache_->pinEvit(task.cluster_id);
                    transfer_task_que_.enqueue(task);
                    wait_transfer_ques_[task.cluster_id].push(task);
                    num_transfering_task_++;
                } else {
                    wait_page_task_que_.enqueue(task);
                    break;
                }
            }
        }
    }
}


}

}
