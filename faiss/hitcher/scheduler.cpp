#include <faiss/hitcher/scheduler.h>

namespace faiss{

namespace hitcher {

Scheduler::Scheduler(StoragePtr storage_ptr, WorkerPtr cpu_worker, std::vector<WorkerPtr>& gpu_workers, bool is_rummy)
: storage_ptr_(storage_ptr), cpu_worker_(cpu_worker), gpu_workers_(gpu_workers), stop_(false), is_rummy_(is_rummy) {
    if (gpu_workers_.size()) {
        no_gpu_ = false;
        num_cls_per_gpu_ = (storage_ptr_->num_clusters + gpu_workers_.size() - 1) / gpu_workers_.size();
    }
    scheduler_ = new std::thread(&faiss::hitcher::Scheduler::thread_work, this);
}

Scheduler::~Scheduler() {
    stop_ = true;
    scheduler_->join();
    delete scheduler_;
}

WorkerPtr Scheduler::get_top_search_worker() {
    if (no_gpu_) {
        return cpu_worker_;
    } else {
        // round robin for top search
        return gpu_workers_.at(top_search_counter_++ % gpu_workers_.size());
    }
}

void Scheduler::assign_bottom_tasks(std::vector<Task>& tasks) {
    if (no_gpu_) {
        cpu_worker_->addBottomTask(tasks);
    } else {
        std::vector<std::vector<Task>> task_list;
        task_list.resize(gpu_workers_.size());
        for (auto & t : tasks) {
            int gpu_id;
            if (is_rummy_) {
                gpu_id = storage_ptr_->queries[t.query_id]->top_search_gpu_id;
            } else {
                gpu_id = t.cluster_id / num_cls_per_gpu_;
            }
            t.set_gpu_id(gpu_id);
            task_list.at(gpu_id).emplace_back(t);
        }

        for (int i = 0; i < gpu_workers_.size(); i++) {
            if (task_list.at(i).size()) {
                gpu_workers_.at(i)->addBottomTask(task_list[i]);
            }
        }
    }
}

void Scheduler::thread_work() {
    int task_cnt = 0;
    while (!stop_) {
        idx_t qid;
        while (storage_ptr_->bottom_ready_queries.try_dequeue(qid)) {
            assign_bottom_tasks(storage_ptr_->queries[qid]->tasks);
            // master_worker_->addBottomTask(storage_ptr_->queries[qid]->tasks);
            // task_cnt += storage_ptr_->queries[qid]->tasks.size();
            // printf("dispatch task cnt: %d\n", task_cnt);
        }
        std::vector<idx_t> qids;
        while (storage_ptr_->top_ready_queries.try_dequeue(qid)) {
            qids.push_back(qid);
        }
        if (qids.size() > 0) {
            get_top_search_worker()->addTopTask(qids);
        }
    }
}


}

}
