from llama_index.core.llama_dataset import download_llama_dataset
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from gptcache.embedding import Onnx as EmbeddingOnnx
from gptcache.embedding import SBERT
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import torch

def work_gpu(sentences, start_idx, end_idx, gpu_id, num_gpu):
    with torch.no_grad():
        tokenizer = BertTokenizer.from_pretrained('gaunernst/bert-tiny-uncased')
        model = BertModel.from_pretrained("gaunernst/bert-tiny-uncased").to("cuda:{0}".format(gpu_id)).eval()
        batch = []
        batch_size = 1024
        result = []
        for idx in tqdm(range(start_idx, end_idx)):
            if idx % num_gpu == gpu_id:
                batch.append(sentences[idx])
                if len(batch) == batch_size:
                    encoded_input = tokenizer(batch, return_tensors='pt', max_length=512, padding=True, truncation=True).to("cuda:{0}".format(gpu_id))
                    embeddings = np.array(model(**encoded_input)['pooler_output'].cpu(), dtype="float32")
                    result.append(embeddings)
                    batch.clear()
        if len(batch) > 0:
            encoded_input = tokenizer(batch, return_tensors='pt', max_length=512, padding=True, truncation=True).to("cuda:{0}".format(gpu_id))
            embeddings = np.array(model(**encoded_input)['pooler_output'].cpu(), dtype="float32")
            result.append(embeddings)
            batch.clear()
    return result

if __name__ == '__main__':
    faq_data = load_dataset("clips/mqa", scope="faq", language="en")
    faq_data = faq_data["train"]
    sentences = []
    for sample in tqdm(faq_data):
        text = sample['name']
        sentences.append(text)
    del faq_data
    file_path = '/data/mqa/mqa.fbin'
    num_embed = len(sentences)
    dim = 128
    print("num db: {0}, dim: {1}".format(num_embed, dim))
    total_size = int(num_embed * dim * 4 + 4 + 4)
    mm = np.memmap(file_path, dtype='uint8', mode='w+', shape=(total_size,))
    count_view = np.ndarray((2,), dtype='int32', buffer=mm, offset=0)
    count_view[0] = num_embed
    count_view[1] = dim
    save_cnt = 0
    with mp.Manager() as manager:
        shared_list = manager.list(sentences)
        segment_len = 1000000
        for start_idx in range(0, num_embed, segment_len):
            end_idx = start_idx + segment_len
            if end_idx > num_embed:
                end_idx = num_embed
            num_gpu = 4
            pool = mp.Pool(num_gpu)
            results = []
            results = [pool.apply_async(work_gpu, args=(shared_list, start_idx, end_idx, gpu_id, num_gpu)) for gpu_id in range(num_gpu)]
            results = [p.get() for p in results]
            pool.terminate()
            for worker_id, worker_results in tqdm(enumerate(results)):
                for batch_results in worker_results:
                    offset = save_cnt * dim * 4 + 8
                    floats_view = np.ndarray((batch_results.shape[0], dim), dtype='float32', buffer=mm, offset=offset)
                    floats_view[:] = batch_results[:]
                    save_cnt += batch_results.shape[0]
    mm.flush()
    assert save_cnt == num_embed
    print(save_cnt, save_cnt)
    