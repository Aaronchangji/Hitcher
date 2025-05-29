#ifndef HITCHER_CLIENT
#define HITCHER_CLIENT

#include "omp.h"
#include <faiss/hitcher/storage.h>
#include <faiss/hitcher/common.h>

namespace faiss{

namespace hitcher {

class Client {
public:
  Client(StoragePtr storage_ptr, int offset, int num_query, double qps);

};

}

}

#endif