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

set(FAISS_GPU_SRC
  GpuAutoTune.cpp
  GpuCloner.cpp
  GpuDistance.cu
  GpuIcmEncoder.cu
  GpuIndex.cu
  GpuIndexBinaryFlat.cu
  GpuIndexFlat.cu
  GpuIndexIVF.cu
  GpuIndexIVFFlat.cu
  GpuIndexIVFPQ.cu
  GpuIndexIVFScalarQuantizer.cu
  GpuResources.cpp
  StandardGpuResources.cpp
  impl/BinaryDistance.cu
  impl/BinaryFlatIndex.cu
  impl/BroadcastSum.cu
  impl/Distance.cu
  impl/FlatIndex.cu
  impl/IndexUtils.cu
  impl/IVFAppend.cu
  impl/IVFBase.cu
  impl/IVFFlat.cu
  impl/IVFFlatScan.cu
  impl/IVFInterleaved.cu
  impl/IVFPQ.cu
  impl/IVFUtils.cu
  impl/IVFUtilsSelect1.cu
  impl/IVFUtilsSelect2.cu
  impl/InterleavedCodes.cpp
  impl/L2Norm.cu
  impl/L2Select.cu
  impl/PQScanMultiPassPrecomputed.cu
  impl/RemapIndices.cpp
  impl/VectorResidual.cu
  impl/IcmEncoder.cu
  utils/BlockSelectFloat.cu
  utils/DeviceUtils.cu
  utils/StackDeviceMemory.cpp
  utils/Timer.cpp
  utils/WarpSelectFloat.cu
  utils/blockselect/BlockSelectFloat1.cu
  utils/blockselect/BlockSelectFloat32.cu
  utils/blockselect/BlockSelectFloat64.cu
  utils/blockselect/BlockSelectFloat128.cu
  utils/blockselect/BlockSelectFloat256.cu
  utils/blockselect/BlockSelectFloatF512.cu
  utils/blockselect/BlockSelectFloatF1024.cu
  utils/blockselect/BlockSelectFloatF2048.cu
  utils/blockselect/BlockSelectFloatT512.cu
  utils/blockselect/BlockSelectFloatT1024.cu
  utils/blockselect/BlockSelectFloatT2048.cu
  utils/warpselect/WarpSelectFloat1.cu
  utils/warpselect/WarpSelectFloat32.cu
  utils/warpselect/WarpSelectFloat64.cu
  utils/warpselect/WarpSelectFloat128.cu
  utils/warpselect/WarpSelectFloat256.cu
  utils/warpselect/WarpSelectFloatF512.cu
  utils/warpselect/WarpSelectFloatF1024.cu
  utils/warpselect/WarpSelectFloatF2048.cu
  utils/warpselect/WarpSelectFloatT512.cu
  utils/warpselect/WarpSelectFloatT1024.cu
  utils/warpselect/WarpSelectFloatT2048.cu
  hitcher/h_gpu_utils.cu
  hitcher/gpu_worker.cu
  hitcher/gpu_profile.cu
  # hitcher/rummy_gpu_worker.cu
)

set(FAISS_GPU_HEADERS
  GpuAutoTune.h
  GpuCloner.h
  GpuClonerOptions.h
  GpuDistance.h
  GpuIcmEncoder.h
  GpuFaissAssert.h
  GpuIndex.h
  GpuIndexBinaryFlat.h
  GpuIndexFlat.h
  GpuIndexIVF.h
  GpuIndexIVFFlat.h
  GpuIndexIVFPQ.h
  GpuIndexIVFScalarQuantizer.h
  GpuIndicesOptions.h
  GpuResources.h
  StandardGpuResources.h
  impl/BinaryDistance.cuh
  impl/BinaryFlatIndex.cuh
  impl/BroadcastSum.cuh
  impl/Distance.cuh
  impl/DistanceUtils.cuh
  impl/FlatIndex.cuh
  impl/GeneralDistance.cuh
  impl/GpuScalarQuantizer.cuh
  impl/IndexUtils.h
  impl/IVFAppend.cuh
  impl/IVFBase.cuh
  impl/IVFFlat.cuh
  impl/IVFFlatScan.cuh
  impl/IVFInterleaved.cuh
  impl/IVFPQ.cuh
  impl/IVFUtils.cuh
  impl/InterleavedCodes.h
  impl/L2Norm.cuh
  impl/L2Select.cuh
  impl/PQCodeDistances-inl.cuh
  impl/PQCodeDistances.cuh
  impl/PQCodeLoad.cuh
  impl/PQScanMultiPassNoPrecomputed-inl.cuh
  impl/PQScanMultiPassNoPrecomputed.cuh
  impl/PQScanMultiPassPrecomputed.cuh
  impl/RemapIndices.h
  impl/VectorResidual.cuh
  impl/scan/IVFInterleavedImpl.cuh
  impl/IcmEncoder.cuh
  utils/BlockSelectKernel.cuh
  utils/Comparators.cuh
  utils/ConversionOperators.cuh
  utils/CopyUtils.cuh
  utils/DeviceDefs.cuh
  utils/DeviceTensor-inl.cuh
  utils/DeviceTensor.cuh
  utils/DeviceUtils.h
  utils/DeviceVector.cuh
  utils/Float16.cuh
  utils/HostTensor-inl.cuh
  utils/HostTensor.cuh
  utils/Limits.cuh
  utils/LoadStoreOperators.cuh
  utils/MathOperators.cuh
  utils/MatrixMult-inl.cuh
  utils/MatrixMult.cuh
  utils/MergeNetworkBlock.cuh
  utils/MergeNetworkUtils.cuh
  utils/MergeNetworkWarp.cuh
  utils/NoTypeTensor.cuh
  utils/Pair.cuh
  utils/PtxUtils.cuh
  utils/ReductionOperators.cuh
  utils/Reductions.cuh
  utils/Select.cuh
  utils/StackDeviceMemory.h
  utils/StaticUtils.h
  utils/Tensor-inl.cuh
  utils/Tensor.cuh
  utils/ThrustUtils.cuh
  utils/Timer.h
  utils/Transpose.cuh
  utils/WarpPackedBits.cuh
  utils/WarpSelectKernel.cuh
  utils/WarpShuffles.cuh
  utils/blockselect/BlockSelectImpl.cuh
  utils/warpselect/WarpSelectImpl.cuh
  hitcher/h_gpu_utils.cuh
  hitcher/gpu_worker.cuh
  hitcher/gpu_profile.cuh
  # hitcher/rummy_gpu_worker.cuh
)

function(generate_ivf_interleaved_code)
  set(SUB_CODEC_TYPE
    "faiss::gpu::Codec<0, 1>"
    "faiss::gpu::Codec<1, 1>"
    "faiss::gpu::Codec<2, 1>"
    "faiss::gpu::Codec<3, 1>"
    "faiss::gpu::Codec<4, 1>"
    "faiss::gpu::Codec<5, 1>"
    "faiss::gpu::Codec<6, 1>"
    "faiss::gpu::CodecFloat"
  )

  set(SUB_METRIC_TYPE
    "faiss::gpu::IPDistance"
    "faiss::gpu::L2Distance"
  )

  # Used for SUB_THREADS, SUB_NUM_WARP_Q, SUB_NUM_THREAD_Q
  set(THREADS_AND_WARPS
    "128|1024|8"
    "128|1|1"
    "128|128|3"
    "128|256|4"
    "128|32|2"
    "128|512|8"
    "128|64|3"
    "64|2048|8"
  )

  # Traverse through the Cartesian product of X and Y
  foreach(sub_codec ${SUB_CODEC_TYPE})
  foreach(metric_type ${SUB_METRIC_TYPE})
  foreach(threads_and_warps_str ${THREADS_AND_WARPS})
    string(REPLACE "|" ";" threads_and_warps ${threads_and_warps_str})
    list(GET threads_and_warps 0 sub_threads)
    list(GET threads_and_warps 1 sub_num_warp_q)
    list(GET threads_and_warps 2 sub_num_thread_q)

    # Define the output file name
    set(filename "template_${sub_codec}_${metric_type}_${sub_threads}_${sub_num_warp_q}_${sub_num_thread_q}")
    # Remove illegal characters from filename
    string(REGEX REPLACE "[^A-Za-z0-9_]" "" filename ${filename})
    set(output_file "${CMAKE_CURRENT_BINARY_DIR}/${filename}.cu")

    # Read the template file
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/impl/scan/IVFInterleavedScanKernelTemplate.cu" template_content)

    # Replace the placeholders
    string(REPLACE "SUB_CODEC_TYPE" "${sub_codec}" template_content "${template_content}")
    string(REPLACE "SUB_METRIC_TYPE" "${metric_type}" template_content "${template_content}")
    string(REPLACE "SUB_THREADS" "${sub_threads}" template_content "${template_content}")
    string(REPLACE "SUB_NUM_WARP_Q" "${sub_num_warp_q}" template_content "${template_content}")
    string(REPLACE "SUB_NUM_THREAD_Q" "${sub_num_thread_q}" template_content "${template_content}")

    # Write the modified content to the output file
    file(WRITE "${output_file}" "${template_content}")

    # Add the file to the sources
    list(APPEND FAISS_GPU_SRC "${output_file}")
  endforeach()
  endforeach()
  endforeach()
  # Propagate modified variable to the parent scope
  set(FAISS_GPU_SRC "${FAISS_GPU_SRC}" PARENT_SCOPE)
endfunction()

generate_ivf_interleaved_code()

if(FAISS_ENABLE_RAFT)
  list(APPEND FAISS_GPU_HEADERS
          impl/RaftFlatIndex.cuh
          impl/RaftIVFFlat.cuh
          impl/RaftIVFPQ.cuh
          utils/RaftUtils.h)
  list(APPEND FAISS_GPU_SRC
          impl/RaftFlatIndex.cu
          impl/RaftIVFFlat.cu
          impl/RaftIVFPQ.cu
          utils/RaftUtils.cu)
endif()

add_library(faiss_gpu STATIC ${FAISS_GPU_SRC})
set_target_properties(faiss_gpu PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)
target_include_directories(faiss_gpu PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

if(FAISS_ENABLE_RAFT)
  target_compile_definitions(faiss PUBLIC USE_NVIDIA_RAFT=1)
  target_compile_definitions(faiss_avx2 PUBLIC USE_NVIDIA_RAFT=1)
  target_compile_definitions(faiss_avx512 PUBLIC USE_NVIDIA_RAFT=1)

  # Mark all functions as hidden so that we don't generate
  # global 'public' functions that also exist in libraft.so
  #
  # This ensures that faiss functions will call the local version
  # inside libfaiss.so . This is needed to ensure that things
  # like raft cublas resources are created and used within the same
  # dynamic library + CUDA runtime context which are requirements
  # for valid execution
  #
  # To still allow these classes to be used by consumers, the
  # respective classes/types in the headers are explicitly marked
  # as 'public' so they can be used by consumers
  set_source_files_properties(
    GpuDistance.cu
    StandardGpuResources.cpp
    impl/RaftFlatIndex.cu
    impl/RaftIVFFlat.cu
    impl/RaftIVFPQ.cu
    utils/RaftUtils.cu
    TARGET_DIRECTORY faiss
    PROPERTIES COMPILE_OPTIONS "-fvisibility=hidden")
  target_compile_definitions(faiss_gpu PUBLIC USE_NVIDIA_RAFT=1)
endif()

# Export FAISS_GPU_HEADERS variable to parent scope.
set(FAISS_GPU_HEADERS ${FAISS_GPU_HEADERS} PARENT_SCOPE)

target_link_libraries(faiss PRIVATE  "$<LINK_LIBRARY:WHOLE_ARCHIVE,faiss_gpu>")
target_link_libraries(faiss_avx2 PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,faiss_gpu>")
target_link_libraries(faiss_avx512 PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,faiss_gpu>")

foreach(header ${FAISS_GPU_HEADERS})
  get_filename_component(dir ${header} DIRECTORY )
  install(FILES ${header}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/faiss/gpu/${dir}
  )
endforeach()

# Prepares a host linker script and enables host linker to support
# very large device object files.
# This is what CUDA 11.5+ `nvcc -hls=gen-lcs -aug-hls` would generate
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld"
[=[
SECTIONS
{
.nvFatBinSegment : { *(.nvFatBinSegment) }
__nv_relfatbin : { *(__nv_relfatbin) }
.nv_fatbin : { *(.nv_fatbin) }
}
]=]
)
target_link_options(faiss_gpu PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld")

find_package(CUDAToolkit REQUIRED)
target_link_libraries(faiss_gpu PRIVATE CUDA::cudart CUDA::cublas $<$<BOOL:${FAISS_ENABLE_RAFT}>:raft::raft> $<$<BOOL:${FAISS_ENABLE_RAFT}>:raft::compiled> $<$<BOOL:${FAISS_ENABLE_RAFT}>:nvidia::cutlass::cutlass>)
target_compile_options(faiss_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xfatbin=-compress-all --expt-extended-lambda --expt-relaxed-constexpr>)
