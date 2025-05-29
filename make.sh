cd build
cmake -DBLA_VENDOR=Intel10_64_dyn -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_CUDA_ARCHITECTURES="86" -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` -DCMAKE_BUILD_TYPE=Release ..
make -j benchmark_stream
make -j benchmark_cpu
make -j profile
cd ..