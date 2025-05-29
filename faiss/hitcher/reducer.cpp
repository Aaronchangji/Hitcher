#include <faiss/hitcher/reducer.h>

namespace faiss{

namespace hitcher {
    
Reducer::Reducer(StoragePtr storage_ptr, int k, int num_thread, bool is_rummy)
: storage_ptr_(storage_ptr), k_(k), num_thread_(num_thread), stop_(false), is_rummy_(is_rummy) {
  threads_.resize(num_thread_);
  for (int i=0; i<num_thread_; i++) {
      threads_[i] = new std::thread(&faiss::hitcher::Reducer::thread_work, this, i);
  }
}

Reducer::~Reducer() {
  stop_ = true;
  for (int i=0; i<num_thread_; i++) {
      if (threads_[i] != nullptr) {
          threads_[i]->join();
          delete threads_[i];
          threads_[i] = nullptr;
      }
  }
  threads_.clear();
  storage_ptr_ = nullptr;
}

int Reducer::getNumValid(idx_t cid) {
  int c_size = storage_ptr_->getClusterSize(cid);
  return ((c_size < k_) ? c_size : k_);
}

void Reducer::reduce(PartialResult &partial_result, float *distances, faiss::idx_t *indices) {
  int num_valid = getNumValid(partial_result.task.cluster_id);
  for (size_t i=0; i<num_valid; i++) {
    float dis = partial_result.distances[i];
    if (dis < distances[0]) {
      int64_t id = partial_result.result_indices[i];
      faiss::maxheap_replace_top(k_, distances, indices, dis, id);
    }
  }
}

void Reducer::classifyBatchResult(BatchPartialResult &batch_partial_result) {
  for (int i=0; i<batch_partial_result.tasks.size(); i++) {
    PartialResult partial_result;
    partial_result.task = batch_partial_result.tasks[i];
    // int pos = i * k_;
    // for (int j = pos; j < pos + k_; j++) {
    //     if (batch_partial_result.result_indices[j] >= 0) {
    //         int c_offset = (int)(batch_partial_result.result_indices[j] & 0xffffffff);
    //         idx_t *cluster_id_ptr = storage_ptr_->getClusterIds(partial_result.task.cluster_id);
    //         batch_partial_result.result_indices[j] = cluster_id_ptr[c_offset];
    //     }
    // }
    // partial_result.distances = std::vector<float>(batch_partial_result.distances.data() + pos, batch_partial_result.distances.data() + pos + k_);
    // partial_result.result_indices = std::vector<idx_t>(batch_partial_result.result_indices.data() + pos, batch_partial_result.result_indices.data() + pos + k_);
    // storage_ptr_->addPartialResult(partial_result);
  }
}

void Reducer::thread_work(int thread_id) {
  while (!stop_) {
    Task task;
    if (is_rummy_) {  // batch-in batch-out
      if (storage_ptr_->batch_finish_) {
        auto cur_time = omp_get_wtime();
        while (storage_ptr_->finish_tasks.try_dequeue(task)) {
          int remain = storage_ptr_->queries[task.query_id]->remain_cluster.fetch_sub(1);
          if (remain == 1) {
            storage_ptr_->queries[task.query_id]->finish_time = cur_time;
            storage_ptr_->num_finished_queries++;
          }
        }
        storage_ptr_->batch_finish_ = false;
      }
    } else {
      while (storage_ptr_->finish_tasks.try_dequeue(task)) {
        int remain = storage_ptr_->queries[task.query_id]->remain_cluster.fetch_sub(1);
        if (remain == 1) {
          storage_ptr_->queries[task.query_id]->finish_time = omp_get_wtime();
          storage_ptr_->num_finished_queries++;
        }
      }
    }
    idx_t qid;
    if (storage_ptr_->finish_top_queries.try_dequeue(qid)) {
      for (auto cid : storage_ptr_->queries[qid]->cids) {
        for (auto bid : storage_ptr_->id_map[cid]) {
          storage_ptr_->queries[qid]->tasks.push_back(Task(qid, bid));
        }
      }
      storage_ptr_->queries[qid]->remain_cluster = storage_ptr_->queries[qid]->tasks.size();
      storage_ptr_->bottom_ready_queries.enqueue(qid);
    }
    // faiss::idx_t query_id;
    // PartialResult partial_result;
    // BatchPartialResult batch_partial_result;
    // if (storage_ptr_->query_ready_to_reduce.try_pop(query_id)) {
    //   Query *query = storage_ptr_->queries[query_id];
    //   std::vector<faiss::idx_t> &indices = query->result_indices;
    //   indices.resize(k_);
    //   std::vector<float> &dis = query->distances;
    //   dis.resize(k_);
    //   faiss::MetricType metric_type = storage_ptr_->index->metric_type;
    //   using HeapForIP = faiss::CMin<float, idx_t>;
    //   using HeapForL2 = faiss::CMax<float, idx_t>;
    //   auto init_result = [&](float* simi, faiss::idx_t* idxi) {
    //     if (metric_type == METRIC_INNER_PRODUCT) {
    //       heap_heapify<HeapForIP>(k_, simi, idxi);
    //     } else {
    //       heap_heapify<HeapForL2>(k_, simi, idxi);
    //     }
    //   };
    //   auto reorder_result = [&](float* simi, idx_t* idxi) {
    //     if (metric_type == METRIC_INNER_PRODUCT) {
    //       heap_reorder<HeapForIP>(k_, simi, idxi);
    //     } else {
    //       heap_reorder<HeapForL2>(k_, simi, idxi);
    //     }
    //   };
    //   init_result(dis.data(), indices.data());
    //   PartialResult parital_result;
    //   while (storage_ptr_->query_results_ques[query_id].try_pop(parital_result)) {
    //     // reduce(parital_result, dis.data(), indices.data());
    //   }
    //   reorder_result(dis.data(), indices.data());
    //   storage_ptr_->addFinishQuery(query_id);
    // } else if (storage_ptr_->tmp_batch_results_que.try_pop(batch_partial_result)) {
    //   classifyBatchResult(batch_partial_result);
    // }
  }
}

}

}