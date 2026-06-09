import chromadb

chroma_client = chromadb.Client()

collection_name = "test_collection"

collection = chroma_client.get_or_create_collection(name=collection_name)

documents = [{"id": "doc1", "text": "Hello world."},
             {"id": "doc2", "text": "How are you?"},
             {"id": "doc3", "text": "what is our name?"},
             {"id": "doc4", "text": "good bye see you later."},
             {"id": "doc5", "text": "This is the fifth document."}]


#define text document
for doc in documents:
    collection.upsert(ids=[doc["id"]], documents=[doc["text"]])

query = "hello world"

result = collection.query(query_texts=[query], 
                          n_results=3)

# print(result)
def similarity_search_with_score(query, k=3):

    
    result = collection.query(query_texts=[query], n_results=k)
    return result
