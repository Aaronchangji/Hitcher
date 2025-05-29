#include <faiss/hitcher/test.h>
#include <faiss/hitcher/storage.h>
#include <faiss/hitcher/worker.h>
#include <faiss/hitcher/scheduler.h>
#include <faiss/hitcher/reducer.h>
#include <faiss/gpu/hitcher/gpu_worker.cuh>
#include <faiss/hitcher/configure.h>
#include <faiss/hitcher/client.h>

#include <future>
#include <iostream>

void showResult(std::vector<faiss::idx_t> &I) {
    for (int i=0; i<I.size(); i++) {
        printf("%ld ", I[i]);
    }
    printf("\n");
}

bool isTheSame(std::vector<faiss::idx_t> &I1, std::vector<faiss::idx_t> &I2) {
    if (I1.size() != I2.size()) {
        return false;
    }
    int num = I1.size();
    for (int i=0; i<num; i++) {
        if (I1[i] != I2[i]) {
            return false;
        }
    }
    return true;
}

void checkResults(faiss::hitcher::StoragePtr storage_ptr, int num_query, int topk) {
    std::vector<float> D(topk);
    std::vector<faiss::idx_t> I(topk);
    for (int query_id=0; query_id<num_query; query_id++) {
        float *query_vec = storage_ptr->getQueryVecFloat32(query_id);
        storage_ptr->index->search(1, query_vec, topk, D.data(), I.data());
        if (!isTheSame(I, storage_ptr->queries[query_id]->result_indices)) {
            printf("result check error on query: %d\n", query_id);
            showResult(I);
            showResult(storage_ptr->queries[query_id]->result_indices);
            return;
        }
        printf("result for query: %d check successful!\n", query_id);
    }
    printf("all result check successful!\n");
    return;
}

int main(int argc, char *argv[]) {
    Configure config(argv[1]);
    std::string index_file = config.index_file;
    std::string query_file = config.query_file;
    int topk = config.topk;
    int nprobe = config.nprobe;
    int num_warmup = config.num_warmup;
    int num_queries = config.num_queries;
    double qps = config.qps;
    int batch_size = config.batch_size;
    int max_bottom_batch_size = config.max_bottom_batch_size;
    int max_pack_num = config.max_pack_num;
    int num_cpu_threads = config.num_cpu_threads;
    double top_batch_threshold = 0.01;
    std::string kernel_mode = config.kernel_mode;
    int transfer_batch_size = config.max_transfering_batch_size;
    bool cpu_offload = config.cpu_offload;
    faiss::hitcher::StoragePtr storage_ptr = std::make_shared<faiss::hitcher::Storage>(config);
    std::cout<<"finish storage init"<<std::endl;
    faiss::hitcher::Reducer reducer(storage_ptr, topk, 4, false);
    std::vector<faiss::hitcher::WorkerPtr> gpu_workers;
    // for (int i = 0; i < config.num_gpu; i++) {
    //     gpu_workers[i] = 
    //     gpu_workers.emplace_back(std::make_shared<faiss::hitcher::GPUWorker>(
    //         storage_ptr, topk, i, batch_size, max_bottom_batch_size, max_pack_num, 
    //         top_batch_threshold, kernel_mode, transfer_batch_size
    //     ));
    //     gpu_workers.at(i)->launchWorker();
    // }
    std::vector<std::future<faiss::hitcher::WorkerPtr>> gpu_workers_future;
    for (int i = 0; i < config.num_gpu; i++) {
        gpu_workers_future.emplace_back(
            std::async(std::launch::async, [&, i]() -> faiss::hitcher::WorkerPtr {
                auto ptr = std::make_shared<faiss::hitcher::GPUWorker>(
                    storage_ptr, topk, i, batch_size, max_bottom_batch_size, max_pack_num, 
                    top_batch_threshold, kernel_mode, transfer_batch_size);
                return std::static_pointer_cast<faiss::hitcher::Worker>(ptr);
            })
        );
    }
    for (int i = 0; i < config.num_gpu; i++) {
        gpu_workers.emplace_back(gpu_workers_future[i].get());
        gpu_workers.at(i)->launchWorker();
    }
    // warmup gpu
    {
        faiss::hitcher::Scheduler scheduler(storage_ptr, nullptr, gpu_workers, false);
        printf("start warmup\n");
        faiss::hitcher::Client(storage_ptr, 0, num_warmup, qps);
        while (storage_ptr->num_finished_queries < num_warmup);
        printf("reset\n");
        for (auto & w : gpu_workers) {
            w->num_finish_task_ = 0;
            w->num_hit_task_ = 0;
        }
        printf("finish warmup\n");
    }
    faiss::hitcher::WorkerPtr cpu_worker = std::make_shared<faiss::hitcher::CPUWorker>(
        storage_ptr, topk, num_cpu_threads, cpu_offload, gpu_workers
    );
    {
        faiss::hitcher::Scheduler scheduler(storage_ptr, cpu_worker, gpu_workers, false);
        faiss::hitcher::Client(storage_ptr, num_warmup, num_queries, qps);
        while (storage_ptr->num_finished_queries < num_warmup + num_queries);
        storage_ptr->getStats(num_warmup, num_warmup + num_queries, config.num_gpu);
        for (int i = 0; i < config.num_gpu; i++) {
            gpu_workers.at(i)->getStats();
        }
        // std::string filename = "logs/batch_trace";
        // std::ofstream file(filename.c_str());
        // for (int i=0; i<batch_times.size(); i++) {
        //     file<<batch_times[i]<<" ";
        //     for (int wid = 0; wid < batch_assign_cnts[i].size(); wid++) {
        //         file<<batch_assign_cnts[i][wid]<<" ";
        //     }
        //     file<<"\n";
        // }
    }
    return 0;
}