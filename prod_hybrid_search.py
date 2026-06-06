from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


TECH_DOCS = [
    Document(
        page_content="Python is a high-level programming language book and name is Python for hacker and programmer known for its simplicity and Product id : SQAPP27, and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python is widely used in web development, data science, artificial intelligence, and automation.",
        metadata={
            "topic": "programming",
            "language": "python",
            "difficulty": "beginner",
        },
    ),
    Document(
        page_content="JavaScript is the language of the web. It runs in browsers and on servers with Node.js. Modern frameworks like React, Vue, and Angular make building interactive web applications efficient. JavaScript supports asynchronous programming with Promises and async/await.",
        metadata={
            "topic": "programming",
            "language": "javascript",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="Machine learning is a subset of AI that enables systems to learn from data. Supervised learning uses labeled data, while unsupervised learning finds patterns in unlabeled data. Popular ML frameworks include TensorFlow, PyTorch, and scikit-learn.",
        metadata={
            "topic": "ai",
            "subtopic": "machine_learning",
            "difficulty": "advanced",
        },
    ),
    Document(
        page_content="LangChain is a framework for building LLM applications. It provides tools for prompts, chains, agents, and memory. LangChain supports multiple LLM providers including OpenAI, Anthropic, and local models.",
        metadata={
            "topic": "ai",
            "subtopic": "llm_frameworks",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs. Key features include state management, cycles and loops, human-in-the-loop workflows, and persistence. LangGraph extends LangChain for complex agent architectures.",
        metadata={
            "topic": "ai",
            "subtopic": "llm_frameworks",
            "difficulty": "advanced",
        },
    ),
    Document(
        page_content="Docker is a platform for containerizing applications. Containers package code and dependencies together for consistent deployment. Docker Compose orchestrates multi-container applications. Kubernetes scales Docker containers in production.",
        metadata={
            "topic": "devops",
            "subtopic": "containers",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="PostgreSQL is an advanced open-source relational database. It supports JSON data types, full-text search, and extensions like pgvector for vector similarity search. PostgreSQL is ACID compliant and highly extensible.",
        metadata={
            "topic": "database",
            "type": "relational",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="Vector databases like Pinecone, Chroma, and Qdrant are optimized for storing and searching embeddings. They enable semantic similarity search for RAG applications. Most support metadata filtering and hybrid search combining keywords with vectors.",
        metadata={"topic": "database", "type": "vector", "difficulty": "intermediate"},
    ),
]


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
   
