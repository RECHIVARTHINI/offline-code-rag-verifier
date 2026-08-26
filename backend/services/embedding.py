from sentence_transformers import SentenceTransformer
import numpy as np
model=SentenceTransformer("BAAI/bge-small-en-v1.5")
def generate_embeddings(chunks):
    embedding=[]
    for chunk in chunks:
        emb=model.encode(chunk["content"],normalize_embeddings=True)
        embedding.append(emb)
    return np.array(embedding)
def query_embedding(query):
    return np.array(model.encode(query,normalize_embeddings=True))
