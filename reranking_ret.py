import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.retrievers.document_compressors import cohere_rerank
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from langsmith import traceable
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
from torch import embedding
from data import documents
from tech_document import TECH_DOCS

load_dotenv()
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma.from_documents(documents=TECH_DOCS, 
                                     embedding=embedding_model,
                                     collection_name = 'cohere')

vector_retriever = vector_store.as_retriever()


api = os.getenv("COHERE_API")
compressor = cohere_rerank.CohereRerank(cohere_api_key=api, top_n=3)

compressor_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_retriever
)

llm = init_chat_model("google_genai:gemini-2.5-flash")

query = 'what is tool calling'

# compress_ret = compressor_retriever.invoke(query)



# for doc in compress_ret:
#     print(doc.page_content)


from data import documents

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)

pairs = [(query, doc) for doc in documents]

scores = reranker.predict(pairs)

for doc, score in zip(documents,scores):
    print(score, doc)
