from fastapi import FastAPI,UploadFile,File
from pydantic import BaseModel
from backend.services.git_service import clone_repo
from backend.services.scanner import scanner
from backend.services.chunker import chunker
from backend.services.embedding import generate_embeddings,query_embedding
from backend.services.storage import create_collection,store_chunks,check_collection,search_query
from backend.services.reranker import rerank_results
app=FastAPI()
class repo_request(BaseModel):
    repo_url:str
class user_query(BaseModel):
    query:str
@app.get('/')
def home():
    return {"Message":"FastAPI is running"}
@app.post('/api/v1/userquery')
def query(request:user_query):
    req=request.query
    embedding=query_embedding(req)
    results=search_query(embedding)
    rerank=rerank_results(req, results)
    if rerank:
        top = rerank[0]
        payload = top["point"].payload
        print(payload.get("symbol"))
        return {"results": [
            {
                "symbol": item["point"].payload.get("symbol"),
                "file_path": item["point"].payload.get("file_path"),
                "score": item["score"],
                "content": item["point"].payload.get("content"),
            }
            for item in rerank[:5]
        ]}
    return {"results": []}
@app.post('/api/v1/upload',status_code=201)
def upload_file(request:repo_request):
    repo_url=request.repo_url
    temp_path=clone_repo(repo_url)
    documents=scanner(temp_path)
    all_chunks=[]
    for document in documents:
        chunks=chunker(document)
        all_chunks.extend(chunks)
    embeddings=generate_embeddings(all_chunks)
    create_collection()
    store_chunks(all_chunks,embeddings)
    check_collection()
    return{"msg":"Repo cloned and Scanned successfully"}
