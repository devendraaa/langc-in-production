import os
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
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
from torch import chunk

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
# embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

document = """
    # API Integration Guide

## Webhooks

Webhooks allow applications to receive real-time notifications when specific events occur. Instead of continuously polling an API for updates, a webhook sends an HTTP POST request to a configured endpoint whenever an event is triggered.

For example, a payment gateway may send a webhook notification when a payment is successfully completed. The receiving server must validate the request, process the payload, and return an appropriate HTTP response code. To ensure reliability, webhook providers often retry failed deliveries multiple times using an exponential backoff strategy.

Webhook security is important. Incoming requests should be verified using signatures, shared secrets, or cryptographic hashes. Logging webhook events helps diagnose delivery failures and monitor system behavior.

## Error Handling

Error handling is a critical aspect of API integration. Applications should gracefully handle network failures, invalid requests, authentication problems, and unexpected server errors.

Common HTTP status codes include 400 Bad Request, 401 Unauthorized, 403 Forbidden, 404 Not Found, 429 Too Many Requests, and 500 Internal Server Error. Each error type should trigger an appropriate recovery strategy.

Retry mechanisms should be implemented carefully. Temporary failures such as timeouts or service unavailability can often be resolved through retries. Permanent failures such as malformed requests should be logged and corrected rather than repeatedly retried.

Monitoring and alerting systems help detect recurring failures and improve application reliability.

## Rate Limiting

Rate limiting protects services from abuse and excessive traffic. API providers often restrict the number of requests that a client can make within a specified time window.

Common rate limiting strategies include fixed windows, sliding windows, token buckets, and leaky buckets. When a client exceeds the allowed request quota, the server typically returns an HTTP 429 Too Many Requests response.

Clients should inspect rate limit headers such as X-RateLimit-Limit, X-RateLimit-Remaining, and Retry-After. These headers provide guidance on how many requests remain and when additional requests can be sent.

Implementing request throttling and exponential backoff helps applications remain compliant with API limits and avoid service disruptions.

## OAuth 2.0 Authentication

OAuth 2.0 is a widely used authorization framework that enables third-party applications to access protected resources on behalf of users.

The framework defines several actors including the resource owner, client application, authorization server, and resource server. Common OAuth flows include Authorization Code Flow, Client Credentials Flow, Device Code Flow, and Refresh Token Flow.

During authentication, the client obtains an access token from the authorization server. This token is then included in API requests using the Authorization header. Access tokens typically expire after a defined period, requiring token refresh operations.

Security best practices include using PKCE, validating redirect URIs, securely storing tokens, and rotating client secrets. Proper OAuth implementation significantly reduces the risk of unauthorized access to protected resources.
"""

def smart_chunkers(
        text:str, use_semantic: bool = True, fallback_chunk_size: int = 500
        ) ->list[str]:
    """Production chunking semantic as primary, fallback to recursive."""

    if use_semantic:
        try:
            semantic_chunker = SemanticChunker(embedding_model,
                                   breakpoint_threshold_type='percentile',

                                   breakpoint_threshold_amount=90)
            
            chunks = semantic_chunker.split_text(text)
            
            max_chunk_size = 2000
            if any(len(c)> max_chunk_size for c in chunks):
                return _recursive_fallback(text, fallback_chunk_size)
            
            return chunks
        
        except Exception as e:
            print(f'semantic chunk failed: {e}, using fallback')
            return _recursive_fallback(text, fallback_chunk_size)
        
        return _recursive_fallback(text, fallback_chunk_size)
    
def _recursive_fallback(text:str, fallback_chunk_size: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=fallback_chunk_size,
                                              chunk_overlap=50,
                                              separators=["\n\n", "\n", " ", ""])
    
    return splitter.split_text(text)

chunks = smart_chunkers(document, use_semantic=True)

print("*" * 40, "smart chunker output", "*" * 40)

print(f"Smart chunk total length:{len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\n--- chunk {i+1} ({len(chunk)}) chars) ---")
    print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
