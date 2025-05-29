#ifndef CONFIGURE_H
#define CONFIGURE_H

#include "inipp.h"
#include <string>
#include <fstream>
#include <vector>

struct Configure {
    std::string index_file;
    std::string id_file;
    std::string vec_file;
    std::string query_file;
    std::string dtype;
    int topk;
    int nprobe;
    int num_warmup;
    int num_queries;
    double qps;
    int batch_size;
    int max_bottom_batch_size;
    int max_pack_num;
    int max_transfering_batch_size;
    int num_cpu_threads;
    int num_gpu;
    double max_schedule_wait_time;
    std::string kernel_mode;
    std::string select_policy;
    std::string cpu_profile_file;
    std::string gpu_profile_file;
    std::string comm_profile_file;
    int cpu_offload;

  
    Configure() {}

    Configure(std::string configure_file) {
        inipp::Ini<char> ini;
        std::ifstream is(configure_file);
        ini.parse(is);
        inipp::get_value(ini.sections["DEFAULT"], "index_file", index_file);
        inipp::get_value(ini.sections["DEFAULT"], "id_file", id_file);
        inipp::get_value(ini.sections["DEFAULT"], "vec_file", vec_file);
        inipp::get_value(ini.sections["DEFAULT"], "query_file", query_file);
        inipp::get_value(ini.sections["DEFAULT"], "dtype", dtype);
        inipp::get_value(ini.sections["DEFAULT"], "topk", topk);
        inipp::get_value(ini.sections["DEFAULT"], "nprobe", nprobe);
        inipp::get_value(ini.sections["DEFAULT"], "num_warmup", num_warmup);
        inipp::get_value(ini.sections["DEFAULT"], "num_queries", num_queries);
        inipp::get_value(ini.sections["DEFAULT"], "qps", qps);
        inipp::get_value(ini.sections["DEFAULT"], "max_schedule_wait_time", max_schedule_wait_time);
        inipp::get_value(ini.sections["DEFAULT"], "max_transfering_batch_size", max_transfering_batch_size);
        inipp::get_value(ini.sections["DEFAULT"], "batch_size", batch_size);
        inipp::get_value(ini.sections["DEFAULT"], "max_bottom_batch_size", max_bottom_batch_size);
        inipp::get_value(ini.sections["DEFAULT"], "max_pack_num", max_pack_num);
        inipp::get_value(ini.sections["DEFAULT"], "num_cpu_threads", num_cpu_threads);
        inipp::get_value(ini.sections["DEFAULT"], "num_gpu", num_gpu);
        inipp::get_value(ini.sections["DEFAULT"], "kernel_mode", kernel_mode);
        inipp::get_value(ini.sections["DEFAULT"], "select_policy", select_policy);
        inipp::get_value(ini.sections["DEFAULT"], "cpu_offload", cpu_offload);
        inipp::get_value(ini.sections["DEFAULT"], "cpu_profile_file", cpu_profile_file);
        inipp::get_value(ini.sections["DEFAULT"], "gpu_profile_file", gpu_profile_file);
        inipp::get_value(ini.sections["DEFAULT"], "comm_profile_file", comm_profile_file);
        // ini.generate(std::cout);
    }
};

#endif