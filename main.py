from qdrant_helpers import run_query, list_all_records, delete_by_source, add_document, get_chat_history
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

app = FastAPI()
handler = Mangum(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"status": "API is running"}


@app.get("/query/{collection_name}/{chat_id}")
def qdrant_query(collection_name: str, chat_id: str, text: str):
    return {"content": run_query(collection_name, chat_id, text)}


@app.get("/chats/{chat_id}")
def chat_history(chat_id: str):
    return {"content": get_chat_history(chat_id)}


@app.get("/documents/{collection_name}")
def list_docs(collection_name: str):
    return {"content": list_all_records(collection_name)}


@app.post("/documents/{collection_name}")
def add_docs(collection_name: str, doc: str):
    return {"content": add_document(collection_name, doc)}


@app.delete("/documents/{collection_name}")
def delete_docs_by_source(collection_name:str, source: str):
    return {"content": delete_by_source(collection_name, source)}

