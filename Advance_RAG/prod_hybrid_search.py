from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from new_document.tech_document import TECH_DOCS

from dotenv import load_dotenv

load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")




vector_store = Chroma.from_documents(documents=TECH_DOCS, 
                                     embedding=embedding_model,
                                     collection_name = 'hybrid_search')


vector_retriever = vector_store.as_retriever(search_kwargs={"k": 2})
print("vector_retriever applied")

bm25_retriever = BM25Retriever.from_documents(TECH_DOCS, k=3)
print("bm25 retriever applied")

ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, 
                                                   bm25_retriever], 
                                                   weights=[0.5, 0.5])
print("hybrid retriever applied")


def test_query(query, name, retriever):
    """Test a query and show results"""
    results = retriever.invoke(query)
    print(f"\n{name} Query:{query}")
    for i, doc in enumerate(results[:3]):
        preview = doc.page_content[:80] + '...'
        print(f' {i+1}. {preview}')
    return results

text_query = [
    'SQAPP27 books name',
    'language of the web',
    'want to learn multi actor suggest me the best library',
    'how to become database engineer'
]

for query in text_query:
    print('=' * 80)
    # vector_sear = test_query(query, "Vector", vector_retriever)
    # bm25_sear = test_query(query, "BM25", bm25_retriever)
    # hybrid_sear = test_query(query, "Hybrid", ensemble_retriever)
   
