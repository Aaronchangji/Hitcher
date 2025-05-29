# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import faiss
import time
import sys
import torch
import os
from tqdm import tqdm

def load_dataset(base_file, dype):
    n, d = map(int, np.fromfile(base_file, dtype="uint32", count=2))
    xb = np.memmap(base_file, dtype=dype, mode="r", offset=8, shape=(n, d))
    return xb

def sample_data(xb, n, sample_num):
    sample_ids = np.random.choice([i for i in range(n)], sample_num, replace=False)
    sample_ids = np.sort(sample_ids)
    training_data = xb[sample_ids].astype(float)
    return training_data

def build_index(index_file, xb, d, nlist, quantizer, metric, training_data):
    print("training...")
    index = faiss.IndexIVFFlat(quantizer, d, nlist, metric)
    index.train(training_data)
    faiss.write_index(index, index_file)
    print("adding data...")
    # index.add(xb)
    list_vec_map = {}
    for list_id in range(index.nlist):
        list_vec_map[list_id] = []
    nb = xb.shape[0]
    batch_size = 1024
    num_batches = (nb + batch_size - 1) // batch_size
    for batch_id in tqdm(range(num_batches)):
        num_queries = min(batch_size, nb - batch_id * batch_size)
        offset = batch_id * batch_size
        batch_q = xb[offset : offset + num_queries].astype(float)
        _, I = index.quantizer.search(batch_q, 1)
        I = I.reshape(num_queries)
        for idx, list_id in enumerate(I):
            list_vec_map[list_id].append(offset + idx)
    assert index.is_trained
    return index, list_vec_map

def save_dataset(xb, list_vec_map, id_file, vec_file, dype):
    print("saving dataset")
    n, d = xb.shape
    nlist = len(list_vec_map)
    total_size = 1 + nlist + n
    id_file_buffer = np.memmap(id_file, dtype='int64', mode='w+', shape=(total_size,))
    id_file_buffer[0] = n
    offset = 1
    for list_id in range(nlist):
        id_file_buffer[offset] = len(list_vec_map[list_id])
        offset += 1
        for vec_id in list_vec_map[list_id]:
            id_file_buffer[offset] = vec_id
            offset += 1
    id_file_buffer.flush()
    vec_file_buffer = np.memmap(vec_file, dtype=dype, mode='w+', shape=(n, d))
    offset = 0
    for list_id in tqdm(range(nlist)):
        for vec_id in list_vec_map[list_id]:
            vec_file_buffer[offset] = xb[vec_id]
            offset += 1
    vec_file_buffer.flush()

def check_dataset(xb, list_vec_map, id_file, vec_file, dtype):
    print("checking dataset")
    n, d = xb.shape
    nlist = len(list_vec_map)
    total_size = 1 + nlist + n
    id_file_buffer = np.memmap(id_file, dtype='int64', mode='r', shape=(total_size,))
    assert id_file_buffer[0] == n
    offset = 1
    for list_id in range(nlist):
        assert id_file_buffer[offset] == len(list_vec_map[list_id])
        offset += 1
        for vec_id in list_vec_map[list_id]:
            assert id_file_buffer[offset] == vec_id
            offset += 1
    vec_file_buffer = np.memmap(vec_file, dtype=dtype, mode='r', shape=(n, d))
    sample_ids = np.random.choice([i for i in range(n)], 10000, replace=False)
    vec_pos_map = {}
    pos = 0
    for list_id in range(nlist):
        for vec_id in list_vec_map[list_id]:
            vec_pos_map[vec_id] = pos
            pos += 1
    for vec_id in sample_ids:
        for dim in range(d):
            assert vec_file_buffer[vec_pos_map[vec_id]][dim] == xb[vec_id][dim]

if __name__ == "__main__":
    dataset = sys.argv[1]
    if dataset == "turing":
        base_file = "/data/turing/base1b.fbin.sample_nb_100000000"
        index_file = "/data/turing/base.fbin.index"
        id_file = "/data/turing/base.fbin.id"
        vec_file = "/data/turing/base.fbin.vec"
        dtype = "float32"
        metric = faiss.METRIC_L2
    elif dataset == "wiki":
        base_file = "/data/wiki/wikipedia.fbin"
        index_file = "/data/wiki/base.fbin.index"
        id_file = "/data/wiki/base.fbin.id"
        vec_file = "/data/wiki/base.fbin.vec"
        dtype = "float32"
        metric = faiss.METRIC_INNER_PRODUCT
    elif dataset == "mqa":
        base_file = "/data/mqa/mqa.fbin"
        index_file = "/data/mqa/base.fbin.index"
        id_file = "/data/mqa/base.fbin.id"
        vec_file = "/data/mqa/base.fbin.vec"
        dtype = "float32"
        metric = faiss.METRIC_INNER_PRODUCT
    else:
        assert False
    xb = load_dataset(base_file, dtype)
    nb, d = xb.shape
    nlist = int(np.sqrt(nb))
    print(dataset, " nb: {0}, d: {1}, nlist: {2}".format(nb, d, nlist))
    faiss.omp_set_num_threads(24)
    # training_data = sample_data(xb, nb, num_sample)
    if metric == faiss.METRIC_INNER_PRODUCT:
        quantizer = faiss.IndexFlatIP(d)
    elif metric == faiss.METRIC_L2:
        quantizer = faiss.IndexFlatL2(d)
    else:
        assert False
    num_sample = int(0.1 * nb)
    training_data = sample_data(xb, nb, num_sample)
    index, list_vec_map = build_index(index_file, xb, d, nlist, quantizer, metric, training_data)
    save_dataset(xb, list_vec_map, id_file, vec_file, dtype)
    check_dataset(xb, list_vec_map, id_file, vec_file, dtype)
