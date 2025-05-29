find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
find_package(TBB REQUIRED)

add_executable(benchmark_stream EXCLUDE_FROM_ALL benchmark_stream.cpp)
target_link_libraries(benchmark_stream PRIVATE faiss OpenMP::OpenMP_CXX Threads::Threads TBB::tbb)

add_executable(benchmark_cpu EXCLUDE_FROM_ALL benchmark_cpu.cpp)
target_link_libraries(benchmark_cpu PRIVATE faiss OpenMP::OpenMP_CXX Threads::Threads TBB::tbb)

add_executable(benchmark_rummy EXCLUDE_FROM_ALL benchmark_rummy.cpp)
target_link_libraries(benchmark_rummy PRIVATE faiss OpenMP::OpenMP_CXX Threads::Threads TBB::tbb)

add_executable(profile EXCLUDE_FROM_ALL profile.cpp)
target_link_libraries(profile PRIVATE faiss OpenMP::OpenMP_CXX Threads::Threads TBB::tbb)
