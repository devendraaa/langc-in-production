from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

from dotenv import load_dotenv

load_dotenv()
from langchain_core.documents import Document

documents = [

    Document(
        page_content="""
        Acute Coronary Syndrome (ACS) commonly presents with chest pain,
        sweating, nausea, shortness of breath and pain radiating to the left arm.
        ECG and Troponin are important diagnostic tests.
        """,
        metadata={
            "department": "Cardiology",
            "source": "acs_guidelines.pdf"
        }
    ),

    Document(
        page_content="""
        Myocardial Infarction (Heart Attack) is a medical emergency caused by
        reduced blood flow to the heart muscle. Common symptoms include chest
        pressure, sweating and breathlessness.
        """,
        metadata={
            "department": "Cardiology",
            "source": "mi_protocol.pdf"
        }
    ),

    Document(
        page_content="""
        Stroke symptoms include facial drooping, arm weakness, speech difficulty
        and sudden loss of balance. Immediate neurological assessment is required.
        """,
        metadata={
            "department": "Neurology",
            "source": "stroke_protocol.pdf"
        }
    ),

    Document(
        page_content="""
        FAST assessment is used to identify stroke:
        Face drooping, Arm weakness, Speech difficulty and Time to call emergency services.
        """,
        metadata={
            "department": "Neurology",
            "source": "fast_assessment.pdf"
        }
    ),

    Document(
        page_content="""
        Type 2 Diabetes Mellitus is characterized by insulin resistance.
        Common symptoms include excessive thirst, frequent urination and fatigue.
        HbA1c is commonly used for diagnosis.
        """,
        metadata={
            "department": "Endocrinology",
            "source": "diabetes_guidelines.pdf"
        }
    ),

    Document(
        page_content="""
        Hospital Asset ID ECG-MON-1001 refers to a bedside cardiac monitor
        installed in ICU Ward A.
        """,
        metadata={
            "department": "Biomedical",
            "source": "asset_registry.pdf"
        }
    ),

    Document(
        page_content="""
        Equipment Serial Number VENT-ICU-2025-445 belongs to a ventilator
        deployed in Critical Care Unit Bed 12.
        """,
        metadata={
            "department": "Biomedical",
            "source": "equipment_inventory.pdf"
        }
    ),

    Document(
        page_content="""
        Patient Emergency Protocol Code RED-CARD-001 is activated for suspected
        cardiac arrest cases requiring immediate resuscitation.
        """,
        metadata={
            "department": "Emergency",
            "source": "emergency_protocols.pdf"
        }
    ),
]

class Hybrid_search:
    """Production Hybrid Retrival with vector search + BM25 search"""

    def __init__(self, documents: List[Document], bm25_weight: float = 0.5, k: int=4):
        self.bm25_weight = bm25_weight
        self.vector_weight = 1 - bm25_weight
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
        
        def add_document(self, document: List[Document]):
            """Add document to bm25 retrivel because it's doesn't support auto increment data"""
            self.vector_store.add_documents(documents=document)

            all_docs = self.vector_store.get()
            self.bm25_retriever = BM25Retriever.from_documents(
                [Document(page_content=doc) for doc in all_docs['documents']], k=self.k
            )

            
        def vector_search(query):
            results = self.vector_retriever.invoke(query)
            return results
        
        def bm25_search(query):
            results = self.bm25_retriever.invoke(query)
            return results
        
        def hybrid_search(query):
            results = self.ensemble_retriever.invoke(query)
            return results
        
        self.vector_search = vector_search
        self.bm25_search = bm25_search
        self.hybrid_search = hybrid_search
        self.add_document = add_document



query = ["RED-CARD-001"]

hybrid_ser = Hybrid_search(documents)

vector_sear = hybrid_ser.vector_search(query[0])

bm25_sear = hybrid_ser.bm25_search(query[0])

hybrid_sear = hybrid_ser.hybrid_search(query[0])

print("*" *40, "vector search output", "*"*40)

for i, data in enumerate(vector_sear):
    print(f"\n output {i+1}: {data.page_content}")
    if data.metadata:
        print(f"  Metadata: {data.metadata}")

print("*" *40, "bm25 search output", "*"*40)

for data in bm25_sear:
    print(f"\n output: {data.page_content}")
    if data.metadata:
        print(f"  Metadata: {data.metadata}")

print("*" *40, "hybrid search output", "*"*40)

for data in hybrid_sear:
    print(f"\n output: {data.page_content}")
    if data.metadata:
        print(f"  Metadata: {data.metadata}")

# print(f"\n this is output: {vector_sear}")



        

