#include <faiss/hitcher/test.h>
#include <faiss/hitcher/storage.h>
#include <faiss/hitcher/worker.h>
#include <faiss/hitcher/scheduler.h>
#include <faiss/hitcher/reducer.h>
#include <faiss/gpu/hitcher/gpu_worker.cuh>
#include <faiss/hitcher/configure.h>
#include <faiss/hitcher/client.h>

#include <future>

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
    double top_batch_threshold = 0.5;
    std::string kernel_mode = config.kernel_mode;
    int transfer_batch_size = config.max_transfering_batch_size;
    faiss::hitcher::StoragePtr storage_ptr = std::make_shared<faiss::hitcher::Storage>(config);
    std::cout<<"finish storage init"<<std::endl;
    faiss::hitcher::Reducer reducer(storage_ptr, topk, 4, true);

    std::vector<faiss::hitcher::WorkerPtr> gpu_workers;
    // for (int i = 0; i < config.num_gpu; i++) {
    //     gpu_workers.emplace_back(std::make_shared<faiss::hitcher::RummyGPUWorker>(
    //         storage_ptr, topk, i, batch_size, max_bottom_batch_size, max_pack_num, 
    //         top_batch_threshold, kernel_mode, transfer_batch_size
    //     ));
    //     gpu_workers.at(i)->launchWorker();
    // }
    std::vector<std::future<faiss::hitcher::WorkerPtr>> gpu_workers_future;
    for (int i = 0; i < config.num_gpu; i++) {
        gpu_workers_future.emplace_back(
            std::async(std::launch::async, [&, i]() -> faiss::hitcher::WorkerPtr {
                auto ptr = std::make_shared<faiss::hitcher::RummyGPUWorker>(
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
        faiss::hitcher::Scheduler scheduler(storage_ptr, nullptr, gpu_workers, true);
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
    {
        faiss::hitcher::Scheduler scheduler(storage_ptr, nullptr, gpu_workers, true);
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