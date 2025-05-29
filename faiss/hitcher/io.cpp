#include <faiss/hitcher/io.h>

namespace faiss{

namespace hitcher {
    faiss::IndexIVFFlat* readIVFFlatIndex(std::string index_file) {
        faiss::Index* index = nullptr;
        index = faiss::read_index(index_file.c_str());
        faiss::IndexIVFFlat* ivf_index = dynamic_cast<faiss::IndexIVFFlat*>(index);
        index = nullptr;
        return ivf_index;
    }

    void* loadQueryDataset(std::string data_file, int &n, int &d, std::string dtype, bool is_pinned) {
        size_t ele_size = 0;
        if (dtype == "float32") {
            ele_size = 4;
        } else if (dtype == "int8") {
            ele_size = 1;
        } else {
            printf("ele type not supported\n");
        }
        FILE* f = fopen(data_file.c_str(), "r");
        fread(&n, 1, sizeof(int), f);
        fread(&d, 1, sizeof(int), f);
        void *data = nullptr;
        if (is_pinned == false) {
            data = new int8_t[size_t(n) * size_t(d) * ele_size];
        } else {
            cudaError_t status = cudaMallocHost((void **)&data, size_t(n) * size_t(d) * ele_size);
            if (status != cudaSuccess) {
                printf("error allocating pinned host memory\n");
            }
        }
        fread(data, ele_size, size_t(n) * size_t(d), f);
        fclose(f);
        return data;
    }

    std::vector<int> loadHotList(std::string filename) {
        std::vector<int> hot_list;
        std::ifstream infile(filename.c_str(), std::ios::in);
        int cid;
        while (infile>>cid) {
            hot_list.push_back(cid);
        }
        return hot_list;
    }

    void fromDeviceToHost(void *dst, void *src, size_t bytes) {
        cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    }

}

} 