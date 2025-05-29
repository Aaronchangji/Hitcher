#include <faiss/hitcher/test.h>

namespace faiss{

namespace hitcher {
    


    void loadIndex(std::string index_file) {
        faiss::Index* index = nullptr;

        // Read the index from the file
        try {
            index = faiss::read_index(index_file.c_str());
            faiss::IndexIVFFlat* ivf_index = dynamic_cast<faiss::IndexIVFFlat*>(index);
            std::cout << "Load index success" << std::endl;
            int n = ivf_index->ntotal;
            int d = ivf_index->d;
            int nlist = ivf_index->nlist;
            printf("n: %d, d: %d, nlist: %d\n", n, d, nlist);
        } catch (std::exception& e) {
            std::cerr << "Error loading index: " << e.what() << std::endl;
            return;
        }

        // Use the index for searching, etc.
        // Remember to deallocate the index when done
        delete index;
    }

    

}

} 