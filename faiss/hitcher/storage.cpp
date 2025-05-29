#include <faiss/hitcher/storage.h>

namespace faiss{

namespace hitcher {

Storage::Storage(Configure &config)
: num_finished_queries(0), num_clusters(0) {
    index = faiss::hitcher::readIVFFlatIndex(config.index_file);
    faiss::MetricType metric_type = index->metric_type;
    if (metric_type == METRIC_INNER_PRODUCT) {
        printf("using metric type: inner product\n");
    } else {
        printf("using metric type: L2\n");
    }
    printf("index nlist: %d\n", index->nlist);
    int nprobe = config.nprobe;
    index->nprobe = nprobe;
    double tc = omp_get_wtime();
    printf("start loading and spliting\n");
    org_cluster_sizes.resize(index->nlist);
    org_cluster_ids.resize(index->nlist);
    org_cluster_codes_ptrs.resize(index->nlist);
    d = index->d;
    dtype = config.dtype;
    if (dtype == "float32") {
        ele_bytes = 4;
    } else if (dtype == "int8") {
        ele_bytes = 1;
    } else {
        printf("data type not support\n");
    }
    // load id file
    {
        FILE* f = fopen(config.id_file.c_str(), "r");
        fread(&nb, 1, sizeof(idx_t), f);
        printf("load %ld vectors with dim: %ld\n", nb, d);
        idx_t cur_cls_size, vec_id;
        idx_t total_lens = 0;
        for (int i=0; i<index->nlist; i++) {
            fread(&cur_cls_size, 1, sizeof(idx_t), f);
            total_lens += cur_cls_size;
            org_cluster_sizes[i] = cur_cls_size;
            for (int j=0; j<cur_cls_size; j++) {
                fread(&vec_id, 1, sizeof(idx_t), f);
                org_cluster_ids[i].push_back(vec_id);
            }
        }
        fclose(f);
        if (total_lens != nb) {
            printf("id file corputed\n");
        }
    }
    // load vec file
    {
        FILE* f = fopen(config.vec_file.c_str(), "r");
        cudaError_t status = cudaMallocHost((void **)&vec_data, size_t(nb) * size_t(d) * ele_bytes);
        if (status != cudaSuccess) {
            printf("error allocating pinned host memory\n");
        }
        fread(vec_data, ele_bytes, size_t(nb) * size_t(d), f);
        fclose(f);
        idx_t offset = 0;
        idx_t vec_bytes = d * ele_bytes;
        for (int i=0; i<index->nlist; i++) {
            org_cluster_codes_ptrs[i] = (vec_data + offset);
            offset += (org_cluster_sizes[i] * vec_bytes);
        }
        split(vec_bytes);
    }
    tc = omp_get_wtime() - tc;
    printf("finish cluster spliting, time cost: %lf\n", tc);
    // cluster_access_cnt.resize(cluster_sizes.size(), 0);
    for (int i = 0; i < cluster_sizes.size(); i++) cluster_access_cnt.emplace_back(0);
    if (config.dtype == "float32") {
        query_vecs_float32 = (float*)faiss::hitcher::loadQueryDataset(config.query_file, nq, dim, config.dtype, true);
        query_vecs_int8 = nullptr;
    } else if (config.dtype == "int8") {
        query_vecs_int8 = (int8_t*)faiss::hitcher::loadQueryDataset(config.query_file, nq, dim, config.dtype, true);
        cudaError_t status = cudaMallocHost((void **)&query_vecs_float32, size_t(nq) * size_t(d) * sizeof(float));
        if (status != cudaSuccess) {
            printf("error allocating pinned host memory\n");
        }
        for (int i=0; i<nq * d; i++) {
            query_vecs_float32[i] = float(query_vecs_int8[i]);
        }
    } else {
        printf("ele type not supported\n");
    }
    queries.resize(nq);
    for (int i=0; i<nq; i++) {
        queries[i] = new Query();
        queries[i]->query_id = i;
    }
    printf("finish reading %d queries vectors with dim %d\n", nq, dim);
}

Storage::~Storage() {
    for (int i=0; i<queries.size(); i++) {
        delete queries[i];
        queries[i] = nullptr;
    }
    queries.clear();
    delete index;
    cudaFreeHost(vec_data);
    if (query_vecs_float32 != nullptr) {
        cudaFreeHost(query_vecs_float32);
    }
    if (query_vecs_int8 != nullptr) {
        cudaFreeHost(query_vecs_int8);
    }
}

void Storage::split(idx_t vec_bytes) {
    size_t page_size = kNumVecPerCluster * vec_bytes;
    idx_t new_cid = 0;
    for (idx_t org_cid=0; org_cid<index->nlist; org_cid++) {
        int8_t *org_cluster = org_cluster_codes_ptrs[org_cid];
        idx_t *org_ids = org_cluster_ids[org_cid].data();
        size_t org_size = org_cluster_sizes[org_cid];
        int num_split = (org_size + kNumVecPerCluster - 1) / kNumVecPerCluster;
        for (int i=0; i<num_split; i++) {
            id_map[org_cid].push_back(new_cid++);
            size_t new_size = kNumVecPerCluster;
            if (org_size - i * kNumVecPerCluster < kNumVecPerCluster) {
                new_size = org_size - i * kNumVecPerCluster;
            }
            cluster_sizes.push_back(new_size);
            cluster_ids_ptrs.push_back(org_ids + i * kNumVecPerCluster);
            cluster_codes_ptrs.push_back(org_cluster + i * page_size);
        }
        num_clusters += num_split;
    }
    for (int cid=0; cid<new_cid; cid++) {
        if (cluster_sizes[cid] <= 0) {
            printf("cluster %d error\n", cid);
        }
    }
}

float* Storage::getQueryVecFloat32(faiss::idx_t query_id) {
    return (query_vecs_float32 + (query_id * dim));
}

int8_t* Storage::getQueryVecInt8(faiss::idx_t query_id) {
    return (query_vecs_int8 + (query_id * dim));
}

size_t Storage::getClusterSize(faiss::idx_t cluster_id) {
    return cluster_sizes[cluster_id];
}

int8_t* Storage::getClusterCodes(faiss::idx_t cluster_id) {
    return cluster_codes_ptrs[cluster_id];
}

faiss::idx_t* Storage::getClusterIds(faiss::idx_t cluster_id) {
    return cluster_ids_ptrs[cluster_id];
}

void Storage::addQuery(faiss::idx_t query_id) {
    queries[query_id]->arrival_time = omp_get_wtime();
    top_ready_queries.enqueue(query_id);
}

void Storage::getStats(int num_skip, int num_queries, int num_gpu) {
    while (num_finished_queries < num_queries);
    double total_time_cost = queries[num_queries - 1]->finish_time - queries[num_skip]->finish_time;
    std::vector<double> wait_latencies, top_latencies, bottom_latencies, e2e_latencies;
    for (int i=num_skip; i<num_queries; i++) {
        wait_latencies.push_back(queries[i]->top_execution_start_time - queries[i]->arrival_time);
        top_latencies.push_back(queries[i]->finish_top_time - queries[i]->top_execution_start_time);
        bottom_latencies.push_back(queries[i]->finish_time - queries[i]->finish_top_time);
        e2e_latencies.push_back(queries[i]->finish_time - queries[i]->arrival_time);
    }
    if (top_latencies.size() == 0) {
        printf("No results found\n");
        return;
    }
    std::sort(wait_latencies.begin(), wait_latencies.end());
    std::sort(top_latencies.begin(), top_latencies.end());
    std::sort(bottom_latencies.begin(), bottom_latencies.end());
    std::sort(e2e_latencies.begin(), e2e_latencies.end());
    int p50_idx = int(top_latencies.size() * 0.5);
    int p95_idx = int(top_latencies.size() * 0.95);
    int p99_idx = int(top_latencies.size() * 0.99);
    printf("[Storage] throughput: %.2f\n", (num_queries - num_skip) / total_time_cost);
    printf("[Storage] p50 wait latency: %.2f ms\n", wait_latencies[p50_idx] * 1000);
    printf("[Storage] p95 wait latency: %.2f ms\n", wait_latencies[p95_idx] * 1000);
    printf("[Storage] p99 wait latency: %.2f ms\n", wait_latencies[p99_idx] * 1000);
    printf("[Storage] p50 top latency: %.2f ms\n", top_latencies[p50_idx] * 1000);
    printf("[Storage] p95 top latency: %.2f ms\n", top_latencies[p95_idx] * 1000);
    printf("[Storage] p99 top latency: %.2f ms\n", top_latencies[p99_idx] * 1000);
    printf("[Storage] p50 bottom latency: %.2f ms\n", bottom_latencies[p50_idx] * 1000);
    printf("[Storage] p95 bottom latency: %.2f ms\n", bottom_latencies[p95_idx] * 1000);
    printf("[Storage] p99 bottom latency: %.2f ms\n", bottom_latencies[p99_idx] * 1000);
    printf("[Storage] p50 e2e latency: %.2f ms\n", e2e_latencies[p50_idx] * 1000);
    printf("[Storage] p95 e2e latency: %.2f ms\n", e2e_latencies[p95_idx] * 1000);
    printf("[Storage] p99 e2e latency: %.2f ms\n", e2e_latencies[p99_idx] * 1000);

    std::vector<int> tasks_per_gpu;
    if (num_gpu > 0) {
        tasks_per_gpu.resize(num_gpu, 0);
        for (int i = num_skip; i < num_queries; i++) {
            for (auto & t : queries[i]->tasks) {
                if (t.gpu_id >= 0) {
                    tasks_per_gpu[t.gpu_id]++;
                }
            }
        }
        for (int i = 0; i < tasks_per_gpu.size(); i++) {
            printf("[Storage] GPU %d are assigned with %d tasks.\n", i, tasks_per_gpu[i]);
        }
    }

    {
        // analysis batch stats
        std::vector<int> nq, nuq, nc, nuc; // number of query, number of unique query, number of cluster, number of unique cluster
        TaskBatchStats stat;
        int batch_cnt = 0;
        while (task_batch_stats_lists.try_dequeue(stat)) {
            nq.emplace_back(stat.num_queries);
            nuq.emplace_back(stat.num_unique_queries);
            nc.emplace_back(stat.num_cls);
            nuc.emplace_back(stat.num_unique_cls);
            batch_cnt++;
        }
        printf("[Storage] There are %lld batches\n", batch_cnt);
        std::sort(nq.begin(), nq.end());
        std::sort(nuq.begin(), nuq.end());
        std::sort(nc.begin(), nc.end());
        std::sort(nuc.begin(), nuc.end());
        int p50_idx = int(nq.size() * 0.5);
        int p95_idx = int(nq.size() * 0.95);
        int p99_idx = int(nq.size() * 0.99);

        printf("[Storage] Index P50: %d, P95: %d, P99: %d\n", p50_idx, p95_idx, p99_idx);

        printf("[Storage] -----Task Batch Stats-----\n");
        printf("[Storage] p50 NQ: %d \n", nq[p50_idx]);
        printf("[Storage] p95 NQ: %d \n", nq[p95_idx]);
        printf("[Storage] p99 NQ: %d \n", nq[p99_idx]);
        printf("[Storage] p50 NUQ: %d \n", nuq[p50_idx]);
        printf("[Storage] p95 NUQ: %d \n", nuq[p95_idx]);
        printf("[Storage] p99 NUQ: %d \n", nuq[p99_idx]);
        printf("[Storage] p50 NC: %d \n", nc[p50_idx]);
        printf("[Storage] p95 NC: %d \n", nc[p95_idx]);
        printf("[Storage] p99 NC: %d \n", nc[p99_idx]);
        printf("[Storage] p50 NUC: %d \n", nuc[p50_idx]);
        printf("[Storage] p95 NUC: %d \n", nuc[p95_idx]);
        printf("[Storage] p99 NUC: %d \n", nuc[p99_idx]);
    }

    fflush(stdout);
}

void Storage::count_unique_access_cluster_rate(std::vector<faiss::idx_t>& query_batch) {
    std::set<faiss::idx_t> unique_clusters;
    int all_cluster_cnt = 0;
    for (auto & qid : query_batch) {
        auto cids = queries[qid]->cids;
        all_cluster_cnt += cids.size();
        for (auto & cid : cids) {
            unique_clusters.emplace(cid);
        }
    }
    size_t num_unique_cids = unique_clusters.size();
    float unique_rate = float(num_unique_cids) / all_cluster_cnt;
    printf("%d unique clusters in %d clusters, with %f unique rate.\n", num_unique_cids, all_cluster_cnt, unique_rate);
}

}

}