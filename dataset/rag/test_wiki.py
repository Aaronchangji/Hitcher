from llama_index.core.llama_dataset import download_llama_dataset
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from datasets import load_dataset
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

if __name__ == "__main__":
    for i in range(0, 4, 5):
        print(i)