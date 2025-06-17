from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from app.chroma_client import create_collection, return_collection_names, upload_data, delete_collection
import os

# Load env
load_dotenv("resources/.env")
API_PASSWORD = os.getenv("API_PASSWORD")
if API_PASSWORD is None:
    raise RuntimeError("API_PASSWORD is not set. Check resources/.env")

app = FastAPI()

api_key_header = APIKeyHeader(name="Authorization", auto_error=True)

def verify_api_key(api_key: str = Security(api_key_header)):
    print("Authorization received:", api_key)
    print("Expected:", API_PASSWORD)
    if api_key != API_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")

class CreateCollectionRequest(BaseModel):
    name: str
    description: str

class DocumentItem(BaseModel):
    documents: List[str]
    metadata: Optional[List[dict]] = None
    embeddings: Optional[List[List[float]]] = None

@app.get("/")
def root():
    return {"message": "ChromaDB API is running."}

@app.post("/collections/create")
def api_create_collection(payload: CreateCollectionRequest, auth=Depends(verify_api_key)):
    collection = create_collection(payload.name, payload.description)
    return {"message": f"Collection '{collection.name}' created successfully."}

@app.get("/collections")
def api_get_collections():
    return return_collection_names()

@app.post("/collections/{collection_name}/upload")
def api_upload(collection_name: str, data: DocumentItem):
    result = upload_data(collection_name, data.dict())
    if isinstance(result, str) and result.startswith("Collection"):
        raise HTTPException(status_code=404, detail=result)
    return {"message": f"Data uploaded to collection '{collection_name}'."}

@app.delete("/collections/{collection_name}")
def api_delete_collection(collection_name: str, auth=Depends(verify_api_key)):
    result = delete_collection(collection_name)
    return {"message": result}
