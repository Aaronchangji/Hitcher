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

def work_gpu(nodes, idx, index_rage):
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-en-v1.5",
        device="cuda:{0}".format(idx)
    )
    batch = []
    batch_size = 2048
    result = []
    for idx in tqdm(index_rage):
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
    parser = SentenceSplitter(chunk_size=64, chunk_overlap=16)
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
    print("finish: ", len(results))
    nodes = []
    for it in results:
        nodes.extend(it)
    print("total: ", len(nodes))
    pool.terminate()
    file_path = '/data/wiki/wikipedia.fbin'
    total_size = int(len(nodes) * 768 * 4 + 4 + 4)
    num_embed = len(nodes)
    print("total bytes: ", total_size)
    results = []
    with mp.Manager() as manager:
        shared_list = manager.list(nodes)
        num_gpu = 4
        pool = mp.Pool(num_gpu)
        num_text = len(nodes)
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
        nodes.clear()
    mm = np.memmap(file_path, dtype='uint8', mode='w+', shape=(total_size,))
    count_view = np.ndarray((2,), dtype='int32', buffer=mm, offset=0)
    count_view[0] = num_embed
    count_view[1] = 768
    offset = 8
    for worker_id, worker_results in enumerate(results):
        print("start saving results from worker {0}".format(worker_id))
        for batch_results in tqdm(worker_results):
            floats_view = np.ndarray((batch_results.shape[0], 768), dtype='float32', buffer=mm, offset=offset)
            floats_view[:] = batch_results[:]
            offset += batch_results.shape[0] * 768 * 4
    mm.flush()
    # for node in tqdm(nodes):
    #     batch.append(node)
    #     if len(batch) == batch_size:
    #         # embedding = np.array(embed_model.get_text_embedding(node.text), dtype="float32")
    #         embeddings = np.array(embed_model.get_text_embedding_batch(batch), dtype="float32")
    #         # print(embeddings.shape)
    #         floats_view = np.ndarray((len(batch), 384), dtype='float32', buffer=mm, offset=offset)
    #         floats_view[:] = embeddings[:]
    #         offset += 384 * len(batch) * 4
    #         batch.clear()
    # if len(batch) > 0:
    #     embeddings = np.array(embed_model.get_text_embedding_batch(batch), dtype="float32")
    #     # print(embeddings.shape)
    #     floats_view = np.ndarray((len(batch), 384), dtype='float32', buffer=mm, offset=offset)
    #     floats_view[:] = embeddings[:]
    #     offset += 384 * len(batch) * 4
    #     batch.clear()
    # mm.flush()
    
# download and install dependencies for benchmark dataset
# rag_dataset, documents = download_llama_dataset("Llama2PaperDataset", "./data")
# loader = WikipediaReader()
# documents = loader.load_data(pages=['Star Wars Movie'])
# print(documents)
# embed_model = HuggingFaceEmbedding(
#     model_name="BAAI/bge-small-en-v1.5"
#     # model_name="BAAI/bge-base-en-v1.5"
#     # model_name="BAAI/bge-large-en-v1.5"
# )
# parser = SentenceSplitter(chunk_overlap=0, chunk_size=128)
# nodes = parser.get_nodes_from_documents(documents)
# for node in nodes:
#     print(node.text)
#     embeddings = embed_model.get_text_embedding(node.text)
# embeddings = embed_model.get_text_embedding("Hello World!")
# print(len(embeddings))
# print(embeddings[:5])
# print(len(nodes))
# for i in range(2):
#     print(nodes[i].text)
#     embeddings = embed_model.get_text_embedding(nodes[i].text)
#     print(embeddings[:5])