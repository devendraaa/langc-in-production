import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
from torch import embedding
from new_document.data import documents
from new_document.tech_document import TECH_DOCS

load_dotenv()
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_base_vec():
    """create the vector store from knowledge base."""
    return Chroma.from_documents(documents=TECH_DOCS, 
                                     embedding=embedding_model,
                                     collection_name = 'hybrid_search')

def demo_multi_query():
    """Multi-query retriever"""
    print("*" * 80)
    print("Multi-query retriever")
    print("*" * 80)

    vector_store = create_base_vec()
    llm = init_chat_model("google_genai:gemini-2.5-flash")

    retrieval = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        llm=llm)
    
    # print(dir(retrieval))
    
    query = 'which tool i have to use to build ai application'

    print(f"\nOriginal query:{query}")
    print("check info log above for gnerated variations...")

    generated_queries = retrieval.llm_chain.invoke(
    {"question": query}
        )
    
    print("\nGenerated Queries:")
    for i, q in enumerate(generated_queries, 1):
        print(f"{i}. {q}")


    docs = retrieval.invoke(query)

    print(f"Retrieved {len(docs)} unique documents.")
    for i, doc in enumerate(docs):
        print(f"\n{i+1}. [{doc.metadata.get('topic', 'N/A')}] {doc.page_content[:100]}")


if __name__ == "__main__":
    demo_multi_query()