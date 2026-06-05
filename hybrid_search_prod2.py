from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()


class Hybrid_search:
    """Production Hybrid Retrival with vector search + BM25 search"""

    def __init__(self, documents: List[Document], bm25_weight: float = 0.5, k: int=4):
        self.bm25_weight = bm25_weight
        self.vector-weight = 1 - bm25_weight
        self.k = k

        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        self.vector_store = Chroma.from_documents(documents=documents, 
                                                 embedding=self.embedding_model,
                                                 collection_name = 'hybrid_search')
        
        self.vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": self.k})

        self.bm25_retriever = BM25Retriever.from_documents(documents, k=self.k)

        self.ensemble_retriever = EnsembleRetriever(retrievers=[self.vector_retriever, 
                                                               self.bm25_retriever], 
                                                      weights=[self.vector_weight, self.bm25_weight])
        

        

