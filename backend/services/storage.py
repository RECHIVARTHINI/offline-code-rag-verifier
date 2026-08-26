from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams,Distance,PointStruct
client=QdrantClient(path="storage/qdrant")
COLLECTION_NAME="chunks"
def create_collection():
    collections=client.get_collections().collections
    if COLLECTION_NAME not in[c.name for c in collections]:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384,
                                        distance=Distance.COSINE)
        )
def store_chunks(chunks,embeddings):
    points=[]
    for i,chunk in enumerate(chunks):
        points.append(PointStruct(id=i,
                                  vector=embeddings[i].tolist(),
                                  payload=chunk))
    client.upsert(collection_name=COLLECTION_NAME,points=points)

def check_collection():
    info=client.get_collection(COLLECTION_NAME)
    print("Collection:",COLLECTION_NAME)
    print("Points:",info.points_count)
    print("Vector size:",info.config.params.vectors.size)

def search_query(embedding):
    result=client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding.tolist(),
        limit=20,
        with_payload=True
    )
    return result.points
        
