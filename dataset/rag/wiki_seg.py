from llama_index.core.llama_dataset import download_llama_dataset
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from datasets import load_dataset
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

def work(wiki_data, parser, index_rage):
    nodes = []
    for idx in tqdm(index_rage):
        document = Document(text=wiki_data[idx]["text"])
        cur_nodes = parser.get_nodes_from_documents([document])
        for node in cur_nodes:
            nodes.append(node.text)
        del document
        cur_nodes.clear()
    return nodes

def work_gpu(nodes, start_idx, end_idx, gpu_id, num_gpu):
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        device="cuda:{0}".format(gpu_id)
    )
    batch = []
    batch_size = 2048
    result = []
    for idx in tqdm(range(start_idx, end_idx)):
        if idx % num_gpu == gpu_id:
            batch.append(nodes[idx])
            if len(batch) == batch_size:
                embeddings = np.array(embed_model.get_text_embedding_batch(batch), dtype="float32")
                result.append(embeddings)
                batch.clear()
    if len(batch) > 0:
        embeddings = np.array(embed_model.get_text_embedding_batch(batch), dtype="float32")
        result.append(embeddings)
        batch.clear()
    return result

if __name__ == "__main__":
    wiki_data = load_dataset("wikipedia", "20220301.en")
    wiki_data = wiki_data["train"]
    # parser = SentenceSplitter(chunk_size=64, chunk_overlap=16)
    parser = SentenceSplitter(chunk_size=100, chunk_overlap=8)
    num_cores = 36
    pool = mp.Pool(num_cores)
    num_documents = len(wiki_data)
    param_dict = []
    begin = 0
    segment_len = num_documents//num_cores + 1
    for i in range(num_cores):
        end = begin + segment_len
        if end > num_documents:
            end = num_documents
        param_dict.append(range(begin, end))
        begin += segment_len
    results = [pool.apply_async(work, args=(wiki_data, parser, param)) for param in param_dict]
    results = [p.get() for p in results]
    nodes = []
    for it in results:
        nodes.extend(it)
    pool.terminate()
    file_path = '/data/wiki/wikipedia.fbin'
    num_embed = len(nodes)
    dim = 384
    print("num db: {0}, dim: {1}".format(num_embed, dim))
    total_size = int(len(nodes) * dim * 4 + 4 + 4)
    mm = np.memmap(file_path, dtype='uint8', mode='w+', shape=(total_size,))
    count_view = np.ndarray((2,), dtype='int32', buffer=mm, offset=0)
    count_view[0] = num_embed
    count_view[1] = dim
    save_cnt = 0
    with mp.Manager() as manager:
        shared_list = manager.list(nodes)
        segment_len = 1000000
        for start_idx in range(0, num_embed, segment_len):
            end_idx = start_idx + segment_len
            if end_idx > num_embed:
                end_idx = num_embed
            num_gpu = 3
            pool = mp.Pool(num_gpu)
            results = []
            results = [pool.apply_async(work_gpu, args=(shared_list, start_idx, end_idx, gpu_id, num_gpu)) for gpu_id in range(num_gpu)]
            results = [p.get() for p in results]
            pool.terminate()
            # save:
            for worker_id, worker_results in tqdm(enumerate(results)):
                for batch_results in worker_results:
                    offset = save_cnt * dim * 4 + 8
                    floats_view = np.ndarray((batch_results.shape[0], dim), dtype='float32', buffer=mm, offset=offset)
                    floats_view[:] = batch_results[:]
                    save_cnt += batch_results.shape[0]
    mm.flush()
    assert save_cnt == num_embed
    print(save_cnt, save_cnt)