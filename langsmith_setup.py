"""
langsmith setup and observability
Production monitoring for langchain/langsmith
"""
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
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

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


@traceable(name="basic tracing")
def basic_tracing():
    """basic tracing langsmith"""
    
    llm = init_chat_model("google_genai:gemini-2.5-flash")
    prompt = ChatPromptTemplate.from_template(
        "explain topic {topic} in one sentence"
        )
    chain = prompt | llm | StrOutputParser()

    print("Basic Tracing Demo:\n")
    print("running chain with langsmith tracing enable...")

    result = chain.invoke({"topic": "machine learning"})
    print(f"result: {result}\n")

@traceable(name="demo 2", tags=['production', 'summarization'])
def demo_2():

    llm = init_chat_model("google_genai:gemini-2.5-flash")
    prompt = ChatPromptTemplate.from_template("summarize: {text}")
    chain = prompt | llm | StrOutputParser()

    print("Basic Tracing Demo:\n")
    print("running chain with langsmith tracing enable...")

    result = chain.invoke({"text": "langsmith povides observability for langchain"})
    print(f"summarization: {result}\n")

@traceable(name="trace with metadata demo", tags=["metadata","filtering"])
def demo_trace_with_metadata(user_id:str, request_type:str):
    llm = init_chat_model("google_genai:gemini-2.5-flash")
    
    #metadata is automatically captured
    result =  llm.invoke(f"hello from user{user_id}")

    return result.content

if __name__ == "__main__":
    basic_tracing()
    demo_2()
    demo_trace_with_metadata("user123", "greeting")    