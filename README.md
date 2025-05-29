# Hitcher: Efficient GPU-based Vector Search via Cluster-Centric Kernel and Hitch-Ride Ordering

Hitcher is an efficient GPU-based vector query processing via cluster-centric kernel and hitch-ride ordering. Although this repo is extended from Faiss repository, Hitcher implements its own system components from scartch, which is located under `faiss/gpu/hitcher/` and `faiss/hitcher/`.


## Dependency

- OpenMP
- TBB

## Build

```
cd build
cmake -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` -DCMAKE_BUILD_TYPE=Release ..
make -j benchmark_stream
make -j benchmark_rummy
make -j benchmark_cpu
make -j profile
cd ..
```

## Run

```
./build/eval/benchmark_cpu configure/configure.ini  # Run Faiss
./build/eval/benchmark_stream configure/configure.ini  # Run Hitcher
./build/eval/benchmark_rummy configure/configure.ini  # Run rummy
```

## Configuration

all configurable paramters are put into configure/configure.ini. `kernel_mode` is a switch to change between query-centric kernel and cluster-centric kernel.
