#include <faiss/hitcher/cpu_profile.h>

namespace faiss{

namespace hitcher {

DummyStorage::DummyStorage(size_t num_vec, size_t d, int k, faiss::MetricType metric) {
    num_vec_page = num_vec;
    dim = d;
    total_num_pages = 100000000 / num_vec_page;
    id_data.resize(total_num_pages * num_vec_page);
    for (idx_t i=0; i<total_num_pages * num_vec_page; i++) {
        id_data[i] = i;
    }
    cudaMallocHost((void **)&pages_data, total_num_pages * num_vec_page * dim * sizeof(float));
    metric_type = metric;
    topk = k;
}

DummyStorage::~DummyStorage() {
    cudaFreeHost(pages_data);
}

float DummyStorage::computeDistance(const float *x, const float *y) {
    float dis = (metric_type == METRIC_INNER_PRODUCT)
                ? faiss::fvec_inner_product(x, y, dim)
                : faiss::fvec_L2sqr(x, y, dim);
    return dis;
}

void DummyStorage::searchBottom(int thread_id) {
    bindCore(thread_id);
    while (!stop) {
        Task task;
        if (task_que.try_dequeue(task)) {
            float *query_vec = pages_data + task.query_id * dim;
            idx_t *cluster_ids = id_data.data() + task.cluster_id * num_vec_page;
            float *cluster_vecs = pages_data + task.cluster_id * num_vec_page * dim;
            std::vector<float> distances(topk);
            std::vector<faiss::idx_t> indices(topk);
            using HeapForIP = faiss::CMin<float, idx_t>;
            using HeapForL2 = faiss::CMax<float, idx_t>;
            auto init_result = [&](float* simi, faiss::idx_t* idxi) {
                if (metric_type == METRIC_INNER_PRODUCT) {
                    heap_heapify<HeapForIP>(topk, simi, idxi);
                } else {
                    heap_heapify<HeapForL2>(topk, simi, idxi);
                }
            };
            auto reorder_result = [&](float* simi, idx_t* idxi) {
                if (metric_type == METRIC_INNER_PRODUCT) {
                    heap_reorder<HeapForIP>(topk, simi, idxi);
                } else {
                    heap_reorder<HeapForL2>(topk, simi, idxi);
                }
            };
            init_result(distances.data(), indices.data());
            for (size_t i=0; i<num_vec_page; i++) {
                float dis = computeDistance(cluster_vecs, query_vec);
                if (dis < distances[0]) {
                    int64_t id = cluster_ids[i];
                    faiss::maxheap_replace_top(topk, distances.data(), indices.data(), dis, id);
                }
                cluster_vecs += dim;
            }
            num_finish++;
        }
    }
}

void DummyStorage::setThreadNum(int num_thread) {
    stop = true;
    for (auto &worker : workers) {
        worker->join();
        delete worker;
    }
    workers.clear();
    workers.resize(num_thread);
    stop = false;
    for (int i=0; i<num_thread; i++) {
        workers[i] = new std::thread(&faiss::hitcher::DummyStorage::searchBottom, this, i);
    }
}

double DummyStorage::profile(int num_task) {
    num_finish = 0;
    std::vector<Task> tasks(num_task);
    for (int i=0; i<num_task; i++) {
        idx_t qid = random_engine() % num_vec_page;
        idx_t cid = random_engine() % num_vec_page;
        tasks[i] = Task(qid, cid);
    }
    double time = omp_get_wtime();
    task_que.enqueue_bulk(tasks.begin(), num_task);
    while (num_finish < num_task);
    time = omp_get_wtime() - time;
    return time;
}

}

}
