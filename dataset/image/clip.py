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
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModelWithProjection
import urllib
import io

def download_image(url):
    urllib_request = urllib.request.Request(
        url,
        data=None,
        headers={"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"},
    )
    with urllib.request.urlopen(urllib_request, timeout=10) as r:
        img_stream = io.BytesIO(r.read())
    return img_stream

if __name__ == '__main__':
    image_metadata = load_dataset("mlfoundations/datacomp_medium")
    image_metadata = image_metadata['train']
    # model = CLIPVisionModelWithProjection.from_pretrained(
    #     "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    # ).cuda()
    # processor = AutoProcessor.from_pretrained(
    #     "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    # )
    error_cnt = 0
    total_num = 128
    for i in tqdm(range(total_num)):
        image_url = image_metadata[i]['url']
        try:
            # image = Image.open(requests.get(image_url, stream=True).raw)
            # inputs = processor(images=image, return_tensors="pt")
            # outputs = model(**inputs)
            # image_embeds = outputs.image_embeds
            # print(image_embeds.shape)
            image = download_image(image_url)
        except:
            error_cnt += 1
    print("success {0} out of {1}".format(total_num-error_cnt, total_num))
            