#include <faiss/gpu/hitcher/rummy_gpu_worker.cuh>

namespace faiss{

namespace hitcher {

RummyGPUWorker::RummyGPUWorker(StoragePtr storage_ptr, int k, int gpu_id, int max_query_batch_size, int max_bottom_batch_size, int max_pack_num, std::string kernel_mode, int transfer_batch_size) 
: GPUWorker(storage_ptr, k, gpu_id, max_query_batch_size, max_bottom_batch_size, max_pack_num, kernel_mode, transfer_batch_size) {
  num_task_to_dispatch_ = 0;
  num_running_task_ = 0;
}

RummyGPUWorker::~RummyGPUWorker() {}

bool RummyGPUWorker::stealTaskFrom(Task &task) {
  return false;
}

void RummyGPUWorker::makeBatchWork() {
  return;
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
    storage_ptr_->finish_top_queries.enqueue(qid);
    for (auto cid : storage_ptr_->queries[qid]->cids) {
      num_task_to_dispatch_ += storage_ptr_->id_map[cid].size();
      num_running_tasks_ += storage_ptr_->id_map[cid].size();
    }
  }
}

void RummyGPUWorker::selectBatch(BatchInfo &batch) {
  Task task;
  std::unordered_map<faiss::idx_t, std::vector<idx_t> > cid2qids;
  std::unordered_map<idx_t, idx_t> unique_queries;
  int dim = storage_ptr_->index->d;
  batch.unique_query_cnt = 0;
  batch.num_queries = 0;
  batch.num_cls = 0;
  idx_t cur_cid;
  while (ready_cids_.try_dequeue(cur_cid)) {
    while (ready_task_que_per_cls_.try_dequeue_from_producer(ptoks_[cur_cid], task)) {
      batch.tasks.emplace_back(task);
      if (unique_queries.find(task.query_id) == unique_queries.end()) {
        unique_queries[task.query_id] = batch.unique_query_cnt++;
      }
      cid2qids[task.cluster_id].emplace_back(task.query_id);
      ++batch.num_queries;
    }
    if (batch.num_queries >= max_bottom_batch_size_) {
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
}

void RummyGPUWorker::searchWork() {
  num_task_to_dispatch_ = 0;
  num_running_task_ = 0;
  faiss::gpu::DeviceScope scope(gpu_res_.device);
  while (!stop_) {
    std::vector<faiss::idx_t> query_batch;
    faiss::idx_t qid;
    while (query_que_.try_dequeue(qid)) {
      query_batch.push_back(qid);
      if (query_batch.size() >= max_top_batch_size_) {
        break;
      }
    }
    if (query_batch.size() > 0) {
      searchTop(query_batch);
      syncSearchTop2(query_batch);
      printf("finish top batch: %d\n", query_batch.size());
    }
    // sync dispatch
    while (num_task_to_dispatch_ > 0);
    while (num_running_tasks_ > 0) {
      // printf()
      BatchInfo batch;
      selectBatch(batch);
      if (batch.tasks.size() > 0) {
        searchBottomQC(batch);
        syncSearchBottom(batch);
        num_finish_task_ += batch.tasks.size();
        num_running_task_ -= batch.tasks.size();
      }
      num_running_tasks_ -= batch.tasks.size();
    }
  }
}

void RummyGPUWorker::dispatchWork() {
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
            ready_task_que_per_cls_.enqueue(ptoks_[task.cluster_id], task);
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
            ready_task_que_per_cls_.enqueue(ptoks_[task.cluster_id], task);
            ready_cids_.enqueue(task.cluster_id);
            num_hit_task_++;
        } else {
            miss_task_que_.enqueue(task);
        }
        num_task_to_dispatch_--;
    }
    // deal with miss
    while (wait_page_task_que_.try_dequeue(task)) {
        if (!wait_transfer_ques_[task.cluster_id].empty()) {
            gpu_cache_->pin(task.cluster_id);
            wait_transfer_ques_[task.cluster_id].push(task);
        } else if (gpu_cache_->hit(task.cluster_id)) {
            gpu_cache_->pin(task.cluster_id);
            ready_task_que_per_cls_.enqueue(ptoks_[task.cluster_id], task);
            ready_cids_.enqueue(task.cluster_id);
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
            ready_task_que_per_cls_.enqueue(ptoks_[task.cluster_id], task);
            ready_cids_.enqueue(task.cluster_id);
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

