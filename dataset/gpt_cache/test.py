from transformers import BertTokenizer, BertModel
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
import torch

with torch.no_grad():
    tokenizer = BertTokenizer.from_pretrained('gaunernst/bert-tiny-uncased')
    model = BertModel.from_pretrained("gaunernst/bert-tiny-uncased").eval().to("cuda:0")
    text1 = "Replace me by any text you'd like."
    text2 = "Replace me by any text"
    text = [text1, text2]
    encoded_input = tokenizer(text, return_tensors='pt', padding=True).to("cuda:0")
    outputs = np.array(model(**encoded_input)['pooler_output'].cpu(), dtype="float32")
    print(outputs.shape)
    # for embedding in outputs:
    #     embedding = embedding.to("cpu")
    #     print(embedding)
    # output = model(**encoded_input)['pooler_output'].to("cpu")
    # print(output['pooler_output'])
