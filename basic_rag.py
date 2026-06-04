from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
import os
import tempfile



load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
# embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#creating knowledge base
KNOWLEDGE_BASE = ["this is a knowledge base about France. The capital of France is Paris.",
                  "this is a knowledge base about animals. The largest mammal is the blue whale.",
                  "this is a knowledge base about Japan. The capital of Japan is Tokyo.",
                  "this is a knowledge base about the sun. The sun is a star at the center of the solar system.",
                    "this is a knowledge base about the moon. The moon is Earth's only natural satellite.",
                    "this is a knowledge base about programming. Python is a popular programming language.",
                    "this is a knowledge base about space. The Milky Way is the galaxy that contains our solar system.",
                    "this is a knowledge base about history. The Great Wall of China was built to protect against invasions.",
                    "this is a knowledge base about sports. Soccer is the most popular sport in the world."]

def create_kb():
    """create the vector store from knowledge base."""

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    doc = [Document(page_content=texts, 
                    metadata={"source": "langchain_knowledge_base.md"}) for texts in KNOWLEDGE_BASE]
    
    # doc = Document(page_content=KNOWLEDGE_BASE, 
    #                metadata={"source": "langchain_knowledge_base.md"})
    
    chunks = splitter.split_documents(doc)

    vector_store = Chroma.from_documents(documents=chunks, 
                                         embedding=embedding_model,
                                         persist_directory="tempfile.mkdtemp()"
                                        )
    return vector_store

def demo_basic_rag():
    """demo the basic RAG process."""
    vector_store = create_kb()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    llm = init_chat_model("google_genai:gemini-2.5-flash")

    prompt = ChatPromptTemplate.from_template("""Answer the question based only on the following context
                                              {context}

                                              Question: {question}

                                              Answer:
                                              
                                              make sure to answer in a concise way and only use the information from the context. 
                                              If you don't know the answer, say you don't know.)
                                              """)
    # format retrieved documents
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} 
                 | prompt 
                 | llm 
                 | StrOutputParser()
                 )
    
    question = ['What is the capital of France?',
                'What is the largest mammal?',
                'What is the capital of Japan?']
    # for q in question:
    #     answer = rag_chain.invoke({"question": q})
    #     print(f"Question: {q}\nAnswer: {answer}\n")

    for q in question:
        answer = rag_chain.invoke(q)
        print(f"Question: {q}")
        print(f"Answer: {answer}")
        print()

if __name__ == "__main__":
    demo_basic_rag()