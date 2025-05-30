# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

cmake_minimum_required(VERSION 3.24.0 FATAL_ERROR)

set(FAISS_LANGUAGES CXX)

# `python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
set(
  CMAKE_PREFIX_PATH 
  ${CMAKE_PREFIX_PATH}
)

if(FAISS_ENABLE_GPU)
  list(APPEND FAISS_LANGUAGES CUDA)
endif()

if(FAISS_ENABLE_RAFT)
include(cmake/thirdparty/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(faiss)
rapids_cuda_init_architectures(pyfaiss)
rapids_cuda_init_architectures(faiss_c_library)
endif()

project(faiss
  VERSION 1.8.0
  DESCRIPTION "A library for efficient similarity search and clustering of dense vectors."
  HOMEPAGE_URL "https://github.com/facebookresearch/faiss"
  LANGUAGES ${FAISS_LANGUAGES})
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Valid values are "generic", "avx2", "avx512".
option(FAISS_OPT_LEVEL "" "generic")
option(FAISS_ENABLE_GPU "Enable support for GPU indexes." ON)
option(FAISS_ENABLE_RAFT "Enable RAFT for GPU indexes." OFF)
option(FAISS_ENABLE_PYTHON "Build Python extension." ON)
option(FAISS_ENABLE_C_API "Build C API." OFF)

if(FAISS_ENABLE_GPU)
  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  enable_language(CUDA)
endif()

if(FAISS_ENABLE_RAFT AND NOT TARGET raft::raft)
   find_package(raft COMPONENTS compiled distributed)
 endif()

add_subdirectory(faiss)

if(FAISS_ENABLE_GPU)
  add_subdirectory(faiss/gpu)
endif()

if(FAISS_ENABLE_PYTHON)
  add_subdirectory(faiss/python)
endif()

if(FAISS_ENABLE_C_API)
  add_subdirectory(c_api)
endif()

add_subdirectory(demos)
add_subdirectory(benchs)
add_subdirectory(tutorial/cpp)
add_subdirectory(eval)

# CTest must be included in the top level to enable `make test` target.
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)

  if(FAISS_ENABLE_GPU)
    add_subdirectory(faiss/gpu/test)
  endif()
endif()
