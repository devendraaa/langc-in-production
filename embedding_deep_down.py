from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def basic_embedding():
    text = "this is machine learning examples"

    single_embedding = embedding_model.embed_query(text=text)
    print(f"vector dimension :{len(single_embedding)}")
    print(f"first 5 valuse: {single_embedding[:5]}")
    print(f"vector norm:{np.linalg.norm(single_embedding):.4f}")

#batch embedding
def batch_emb():
    text = [" this is machine learning", 
            "this is deep learning", 
            "this is reinforcement learning and i was"]
    
    batch_embedding = embedding_model.embed_documents(text)

    for i, emb in enumerate(batch_embedding):
        print(f"text {i+1}: Vector dimension{len(emb)}")
        print(f"first 5 values:{emb[:5]}")
        print(f"vector norm:{np.linalg.norm(emb):.4f}")
    

def similarity_ser():

    docs = [
        "python is most use language in Ai",
        "java is a popular programming language",
        "c++ is a very fast language",
        "ruby is widely use language in web development",
        "cats are not domestic animals",
        "dogs are most best animals"
    ]

    query = "python programming language exit"

    doc_vector = embedding_model.embed_documents(docs)
    query_vector = embedding_model.embed_query(query)

    def cosin_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    similarities = [cosin_similarity(query_vector, doc_vec) for doc_vec in doc_vector]

    ranked_sim = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)

    print(f"query: {query}\n")
    print(f"Ranked by similarity:")
    for doc, sim in ranked_sim:
        print(f"{doc}: {sim:.4f}")

    





if __name__=="__main__":
    # basic_embedding()
    # batch_emb()
    similarity_ser()


