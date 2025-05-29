#ifndef HITCHER_IO_H
#define HITCHER_IO_H

#include <iostream>
#include <string>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <cuda_runtime.h>
#include <fstream>
#include <ios>
#include <vector>

namespace faiss{

namespace hitcher {
    faiss::IndexIVFFlat* readIVFFlatIndex(std::string index_file);

    void* loadQueryDataset(std::string data_file, int &n, int &d, std::string dtype, bool is_pinned);

    std::vector<int> loadHotList(std::string filename);

    void fromDeviceToHost(void *dst, void *src, size_t bytes);

}

}

#endif