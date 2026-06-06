from bz2 import compress
import os
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers import LLMChainExtractor
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langsmith import traceable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
from torch import chunk
from data import documents
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


def demo_parent_ret():
    """parent document retrieval: small chunk for search and long chunk for context"""

    print("*" * 80)
    print("Parent document retreiver")
    print("small chunk for precise search, long chunk foe context")
    print("*" * 80)

    long_doc = Document(page_content=documents,
                        metadata={"source": "ai_agentic_guide.md"})
    
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=8000,
                                                     chunk_overlap=100)
    
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200,
                                                    chunk_overlap=20)
    
    vector_store = Chroma(collection_name="parent_child_demo",
                          embedding_function=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
                        )
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        metadata_field="source")


    retriever.add_documents([long_doc])

    query = 'what is Langgraph used for?'

    child_doc = vector_store.similarity_search(query=query, k=1)
    print(f"\n-- Child chunk (what search found)---")
    print(f"Length: {len(child_doc[0].page_content)} chars")
    print(f"Content:{child_doc[0].page_content}")

    #parent retrieval get full context
    parent_docs = retriever.invoke(query)
    print(f"\n--- Parent Chunk (what is returned)---")
    print(f"Length: {len(parent_docs[0].page_content)} chars")
    print(f"Content preview: {parent_docs[0].page_content[:300]}...")


def demo_contextual_compression():
    """Contextual compression extracts only relevant parts"""

    print("*" * 80)
    print("Contextual Compression")
    print("*" * 80)

    vector_store = Chroma(collection_name="parent_child_demo",
                          embedding_function=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
                        )
    llm = init_chat_model("google_genai:gemini-2.5-flash")

    compressor = LLMChainExtractor.from_llm(llm)

    compressor_reteiever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vector_store.as_retriever(search_kwarg={"k":4})
    )
    query = "what framework exist for building LLM applications?"

    base_docs = vector_store.as_retriever(search_kwargs={"k":4}).invoke(query)
    print(f"\n--- without compressor (full parent chunks)---")

    for doc in base_docs:
        print(f"Length: {len(doc.page_content)} chars")
        print(f"Content preview: {doc.page_content[:300]}...")

    compressed_doc = compressor_reteiever.invoke(query)
    print(f"\n--- with compressor (only relevant parts)---")
    for doc in compressed_doc:
        print(f"Length: {len(doc.page_content)} chars")
        print(f"Content: {doc.page_content}\n")




if __name__ == "__main__":
    # demo_parent_ret()
    demo_contextual_compression()
    

