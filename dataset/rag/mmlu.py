from llama_index.core.llama_dataset import download_llama_dataset
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    mmlu_data = load_dataset("cais/mmlu", "all")
    print(mmlu_data)
    questions = []
    dataset_tag = ["test", "validation", "dev", "auxiliary_train"]
    # dataset_tag = ["test"]
    for tag in dataset_tag:
        for idx, page in enumerate(mmlu_data[tag]):
            questions.append(page["question"])
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    file_path = '/data/wiki/mmlu.bin'
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

exit(0)

# build basic RAG system
index = VectorStoreIndex.from_documents(documents=documents)
query_engine = index.as_query_engine()

# evaluate using the RagEvaluatorPack
RagEvaluatorPack = download_llama_pack(
    "RagEvaluatorPack", "./rag_evaluator_pack"
)
rag_evaluator_pack = RagEvaluatorPack(
    rag_dataset=rag_dataset, query_engine=query_engine
)

############################################################################
# NOTE: If have a lower tier subscription for OpenAI API like Usage Tier 1 #
# then you'll need to use different batch_size and sleep_time_in_seconds.  #
# For Usage Tier 1, settings that seemed to work well were batch_size=5,   #
# and sleep_time_in_seconds=15 (as of December 2023.)                      #
############################################################################

# benchmark_df = await rag_evaluator_pack.arun(
#     batch_size=20,  # batches the number of openai api calls to make
#     sleep_time_in_seconds=1,  # seconds to sleep before making an api call
# )