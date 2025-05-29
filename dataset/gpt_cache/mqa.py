from llama_index.core.llama_dataset import download_llama_dataset
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from gptcache.embedding import Onnx as EmbeddingOnnx
from gptcache.embedding import SBERT
from datasets import load_dataset
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

def work_gpu(sentences, idx, index_rage):
    embed_model = HuggingFaceEmbedding(
        model_name="gaunernst/bert-tiny-uncased",
        device="cuda:{0}".format(idx)
    )
    batch = []
    batch_size = 4096
    result = []
    for idx in tqdm(index_rage):
        batch.append(sentences[idx])
        if len(batch) == batch_size:
            embeddings = np.array(embed_model.get_text_embedding_batch(batch), dtype="float32")
            result.append(embeddings)
            batch.clear()
    if len(batch) > 0:
        embeddings = np.array(embed_model.get_text_embedding_batch(batch), dtype="float32")
        result.append(embeddings)
        batch.clear()
    return result

if __name__ == '__main__':
    faq_data = load_dataset("clips/mqa", scope="faq", language="en")
    faq_data = faq_data["train"]
    file_path = '/data/mqa/mqa.fbin'
    num_embed = len(faq_data)
    dim = 128
    total_size = int(num_embed * dim * 4 + 4 + 4)
    results = []
    sentences = []
    for sample in tqdm(faq_data):
        sentences.append(sample['name'])
        if len(sentences) == 100000:
            break
    del faq_data
    with mp.Manager() as manager:
        shared_list = manager.list(sentences)
        num_gpu = 4
        pool = mp.Pool(num_gpu)
        num_text = len(sentences)
        param_dict = []
        begin = 0
        segment_len = num_text//num_gpu + 1
        for i in range(num_gpu):
            end = begin + segment_len
            if end > num_text:
                end = num_text
            param_dict.append(range(begin, end))
            begin += segment_len
        results = [pool.apply_async(work_gpu, args=(shared_list, idx, param)) for idx, param in enumerate(param_dict)]
        results = [p.get() for p in results]
        print("finish: ", len(results))
        pool.terminate()
    mm = np.memmap(file_path, dtype='uint8', mode='w+', shape=(total_size,))
    count_view = np.ndarray((2,), dtype='int32', buffer=mm, offset=0)
    count_view[0] = num_embed
    count_view[1] = dim
    offset = 8
    for worker_id, worker_results in tqdm(enumerate(results)):
        print("start saving results from worker {0}".format(worker_id))
        for batch_results in tqdm(worker_results):
            floats_view = np.ndarray((batch_results.shape[0], dim), dtype='float32', buffer=mm, offset=offset)
            floats_view[:] = batch_results[:]
            offset += batch_results.shape[0] * dim * 4
    mm.flush()
    