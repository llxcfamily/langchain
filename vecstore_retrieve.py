from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.merge import MergedDataLoader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter

# step1: load dataset(txt/csv/json/docx/pdf/...)
loader1 = Docx2txtLoader("guide/1/01.docx")
loader2 = Docx2txtLoader("guide/1/02.docx")
loader_all = MergedDataLoader(loaders=[loader1, loader2])
documents = loader_all.load()

# step2: split text to chunk
"""
# differnt textsplitter methods
from transformers import LlamaTokenizerFast
tokenizer = LlamaTokenizerFast.from_pretrained("decapoda-research/llama-7b-hf")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=200, chunk_overlap=0)
"""
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=256, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
print(docs[0:5])


# step3: convert chunks to embedding
from langchain.embeddings import HuggingFaceEmbeddings
"""
# llama.cpp embedding
from langchain.embeddings import LlamaCppEmbeddings
embeddings = LlamaCppEmbeddings(model_path="./llama-2-7b.Q4_0.gguf")
"""

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# step4: link to elasticsearch vecstore
from langchain.vectorstores import ElasticsearchStore
db = ElasticsearchStore.from_documents(
    docs, embeddings, es_url="http://localhost:9200", index_name="test-basic", es_user="elastic", es_password="0dBJpB09cT8xk6Z_Ue6s", 
)
# update index
db.client.indices.refresh(index="test-basic")
print("update index done")

# step5: retrieve
query = "已满十四周岁不满十六周岁的人，犯罪后如何处罚?"
results = db.similarity_search(query, k=3)
print(results)
