#include <faiss/hitcher/cpu_profile.h>
#include <faiss/gpu/hitcher/gpu_profile.cuh>
#include <faiss/hitcher/test.h>
#include <faiss/hitcher/storage.h>
#include <faiss/hitcher/worker.h>
#include <faiss/hitcher/reducer.h>
#include <faiss/gpu/hitcher/gpu_worker.cuh>
#include <faiss/hitcher/configure.h>
#include <unordered_map>

int main(int argc, char *argv[]) {
    int gpu_id = 0;
    size_t num_vec_page = atoi(argv[1]);
    size_t dim = atoi(argv[2]);
    int max_n_cls = atoi(argv[3]);
    Configure config(argv[4]);
    int topk = 10;
    faiss::gpu::StandardGpuResources gpu_res;
    gpu_res.getResources()->initializeForDevice(gpu_id);
    gpu_res.setTempMemory(faiss::hitcher::kGPUTmpMemSizeBottom * 8);
    faiss::hitcher::DummyCache dummy_cache(gpu_id, gpu_res.getDefaultStream(gpu_id), num_vec_page, dim, gpu_res.getResources().get());
    {
        // profile compute
        std::cout<<"profile compute"<<std::endl;
        int n_query = 64;
        int warmup = 40, num_pass = 80;
        std::string gpu_profile_file = config.gpu_profile_file;
        std::ofstream file(gpu_profile_file.c_str());
        int n_cls = max_n_cls;
        // for (int n_cls = 16; n_cls <= max_n_cls; n_cls *= 2) {
            // for (int num_pack = n_cls; num_pack <= n_cls * 8; num_pack *= 2) {
        for (int num_pack = n_cls; num_pack <= n_cls * 16; num_pack += n_cls) {
            std::vector<double> time_costs;
            double total_time = 0;
            for (int i=0; i<num_pass; i++) {
                auto batch = dummy_cache.generateRandomBatch(n_cls, n_query, num_pack, 16, true);
                double time_cost = dummy_cache.executeBatch(batch, gpu_res.getResources().get());
                if (i >= warmup) {
                    total_time += time_cost;
                    time_costs.push_back(time_cost);
                }
            }
            double avg_latency = total_time / time_costs.size();
            file<<n_cls<<" "<<num_pack<<" "<<avg_latency<<"\n";
            printf("n_cls: %d, n_query: %d, num_pack: %d, kernel latency: %lf ms\n", n_cls, n_query, num_pack, avg_latency); 
        }
        // }
    }
    // {
        // profile compute for QC
    //     std::cout<<"profile compute QC"<<std::endl;
    //     int total_n_query = 512, max_probe = 5;
    //     int warmup = 50, num_pass = 100;
        // std::string gpu_profile_file = config.gpu_profile_file;
        // std::ofstream file(gpu_profile_file.c_str());
        // for (int n_cls = 16; n_cls <= max_n_cls; n_cls *= 2) {
        //     std::cout << "profile ncls = " << n_cls << std::endl;
    //     for (int num_query = 1; num_query <= 512; num_query *= 2) {
    //         for (int probe = 1; probe < max_probe; probe ++) {
    //             std::vector<double> time_costs;
    //             double total_time = 0;
    //             for (int i=0; i<num_pass; i++) {
    //                 auto batch = dummy_cache.generateRandomBatchQC(num_query, total_n_query, probe);
    //                 double time_cost = dummy_cache.executeBatchQC(batch, gpu_res.getResources().get(), probe);
    //                 if (i >= warmup) {
    //                     total_time += time_cost;
    //                     time_costs.push_back(time_cost);
    //                 }
    //             }
    //             double avg_latency = total_time / time_costs.size();
    //             printf("n_query: %d, probe: %d, kernel latency: %lf ms\n", num_query, probe, avg_latency); 
    //         }
    //     }
        // }
    // }
    // {
    //     // profile transfer
    //     std::cout<<"profile transfer"<<std::endl;
    //     int warmup = 5, num_pass = 10;
    //     std::string comm_profile_file = config.comm_profile_file;
    //     std::ofstream file(comm_profile_file.c_str());
    //     for (int n_cls = 1; n_cls <= 64; n_cls ++) {
    //         std::vector<double> time_costs;
    //         double total_time = 0;
    //         for (int i=0; i<num_pass; i++) {
    //             double time_cost = dummy_cache.transferBatch(n_cls, gpu_res.getResources().get());
    //             if (i >= warmup) {
    //                 total_time += time_cost;
    //                 time_costs.push_back(time_cost);
    //             }
    //         }
    //         double avg_latency = total_time / time_costs.size();
    //         file<<n_cls<<" "<<avg_latency<<"\n";
    //         printf("n_cls: %d, transfer latency: %lf ms, throughput: %lf\n", n_cls, avg_latency, (n_cls/avg_latency)); 
    //     }
    // }
    // {
    //     // profile cpu worker
    //     faiss::hitcher::DummyStorage dummy_storage(num_vec_page, dim, topk, faiss::MetricType::METRIC_L2);
    //     int max_thread_num = omp_get_max_threads();
    //     std::string cpu_profile_file = config.cpu_profile_file;
    //     std::ofstream file(cpu_profile_file.c_str());
    //     // train
    //     for (int i=1; i<=max_thread_num; i++) {
    //         int pass = 10;
    //         int warmup = 5;
    //         double total_time = 0;
    //         int batch_size = 8 * i;
    //         dummy_storage.setThreadNum(i);
    //         for (int j=0; j<pass; j++) {
    //             double batch_time = dummy_storage.profile(batch_size);
    //             if (j >= warmup) {
    //                 total_time += batch_time;
    //             }
    //         }
    //         double throughput = batch_size * (pass - warmup) * num_vec_page / total_time;
    //         file<<i<<" "<<throughput<<"\n";
    //         printf("thread num: %d, throughput: %lf vec/s\n", i, throughput);
    //     }
    //     dummy_storage.setThreadNum(0);
    // }
    return 0;
}
