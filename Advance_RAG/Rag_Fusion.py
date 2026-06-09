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
# from all_document.data import documents
import os
from langchain_classic.document_loaders import DirectoryLoader, TextLoader, PyMuPDFLoader
from langchain_chroma import Chroma

load_dotenv()

import textwrap

def wrap_text(text, width=90):
    #split the input text into line based on newline characters
    line = text.split('\n')

    #wrap each line individual
    wrapped_line = [textwrap.fill(l, width) for l in line]

    #join wrapper line back together using newline characters
    wrapped_text = '\n'.join(wrapped_line)

    return wrapped_text

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = init_chat_model("google_genai:gemini-2.5-flash")
# result = llm.invoke("what is genai")
# print(result.content)

data_path = "G:\\langc-in-production\\all_document"

loader = DirectoryLoader(path=data_path,
                         show_progress=True,
                         loader_cls=PyMuPDFLoader, 
                         glob="**/*.pdf")
docs = loader.load()

print(len(docs))