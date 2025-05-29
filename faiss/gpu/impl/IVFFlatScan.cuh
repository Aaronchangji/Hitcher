/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/MetricType.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <faiss/gpu/impl/GpuScalarQuantizer.cuh>
#include <faiss/gpu/utils/DeviceVector.cuh>
#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
namespace gpu {

class GpuResources;

void runIVFFlatScan(
        Tensor<float, 2, true>& queries,
        Tensor<idx_t, 2, true>& listIds,
        DeviceVector<void*>& listData,
        DeviceVector<void*>& listIndices,
        IndicesOptions indicesOptions,
        DeviceVector<idx_t>& listLengths,
        idx_t maxListLength,
        int k,
        faiss::MetricType metric,
        bool useResidual,
        Tensor<float, 3, true>& residualBase,
        GpuScalarQuantizer* scalarQ,
        // output
        Tensor<float, 2, true>& outDistances,
        // output
        Tensor<idx_t, 2, true>& outIndices,
        GpuResources* res
);

void runIVFFlatScanHQC(
    Tensor<float, 2, true>& queries,
    Tensor<idx_t, 2, true>& listIds,
    int num_valid,
    DeviceVector<void*>& listData,
    DeviceVector<void*>& listIndices,
    DeviceVector<idx_t>& listLengths,
    idx_t maxListLength,
    int k,
    faiss::MetricType metric,
    Tensor<float, 2, true>& outDistances,
    Tensor<idx_t, 2, true>& outIndices,
    GpuResources* res,
    cudaStream_t stream
);

void runIVFFlatScanHQCCompact(
    Tensor<float, 2, true>& queries,
    Tensor<idx_t, 1, true>& listQueryIds,  
    Tensor<idx_t, 1, true>& listIds,
    DeviceVector<void*>& listData,
    DeviceVector<void*>& listIndices,
    DeviceVector<idx_t>& listLengths,
    idx_t maxListLength,
    int k,
    faiss::MetricType metric,
    Tensor<float, 2, true>& outDistances,
    Tensor<idx_t, 2, true>& outIndices,
    GpuResources* res,
    cudaStream_t stream
);

constexpr size_t kMaxQueryPerList = 8;

void runIVFFlatScanHCC(
    Tensor<float, 2, true>& queryVecs,      // unique query embedding
    Tensor<idx_t, 1, true>& queryLocalIds,  // query idx --> query embedding idx
    Tensor<idx_t, 1, true>& queryOffsets,   // cid --> num of query
    Tensor<idx_t, 1, true>& listIds,        // unique cids
    DeviceVector<void*>& listData,
    DeviceVector<void*>& listIndices,
    DeviceVector<idx_t>& listLengths,
    idx_t maxListLength,
    int k,
    faiss::MetricType metric,
    Tensor<float, 2, true>& outDistances,
    Tensor<idx_t, 2, true>& outIndices,
    GpuResources* res,
    cudaStream_t stream
);

void show(idx_t *d_data, int size, cudaStream_t stream);

} // namespace gpu
} // namespace faiss
