from llama_index.core.llama_dataset import download_llama_dataset
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    google_natural_question_data = load_dataset("rojagtap/natural_questions_clean")
    google_natural_question_data = google_natural_question_data["train"]
    print(google_natural_question_data)
    exit(0)
    questions = []
    for idx, page in enumerate(google_natural_question_data):
        questions.append(page["question"])
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    file_path = '/data/wiki/google_nq.bin'
    total_size = int(len(questions) * 384 * 4 + 4 + 4)
    print("total rows: {0}, total bytes: {1}".format(len(questions), total_size))
    mm = np.memmap(file_path, dtype='uint8', mode='w+', shape=(total_size,))
    count_view = np.ndarray((2,), dtype='int32', buffer=mm, offset=0)
    count_view[0] = len(questions)
    count_view[1] = 384
    offset = 8
    for idx, question in tqdm(enumerate(questions)):
        embedding = np.array(embed_model.get_query_embedding(question), dtype="float32")
        # print(embeddings.shape)
        floats_view = np.ndarray((1, 384), dtype='float32', buffer=mm, offset=offset)
        floats_view[:] = embedding[:]
        offset += 384 * 4
    mm.flush()