import os
from typing import List
from langchain.schema import BaseMessage
from langchain.agents import initialize_agent
from langchain import OpenAI
from langchain.memory import ConversationBufferWindowMemory, RedisChatMessageHistory
from langchain.agents import Tool
import openai
import qdrant_client
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from qdrant_client.http import models


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
QDRANT_HOST = os.environ.get("QDRANT_HOST")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
HUGGINGFACEHUB_API_TOKEN= os.environ.get("HUGGINGFACEHUB_API_TOKEN")
REDIS_CONN = os.environ.get("REDIS_CONN")

embeddings = HuggingFaceHubEmbeddings(
    repo_id="sentence-transformers/all-mpnet-base-v2",
) # type: ignore

def get_client():
    return qdrant_client.QdrantClient(
        url=QDRANT_HOST,
        api_key=QDRANT_API_KEY,
    )


def init_qdrant(collection_name: str) -> Qdrant:
    client = get_client()

    return Qdrant(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings.embed_query
    )


def init_index(collection_name: str):
    print("initializing index")
    qdrant = init_qdrant(collection_name)
    llm=ChatOpenAI(client=openai, temperature=0, openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo')

    return RetrievalQA.from_chain_type(
        retriever=qdrant.as_retriever(),
        llm=llm,
        chain_type="stuff",
    )


index = init_index("my_documents")

tools = [
    Tool(
        name = "Meal Planner",
        func=lambda q: str(index.run(q)), # type: ignore
        description="useful for when you want to answer questions about nutrition. The input to this tool should be a complete english sentence.",
        return_direct=True
    ),
]


def get_chat_history(chat_id: str) -> List[BaseMessage] | str:
    global index
    if index is None:
        return "index is empty"

    if REDIS_CONN is None:
        return "REDIS_CONN is not set"

    history = RedisChatMessageHistory(chat_id, url=REDIS_CONN)
    return history.messages


def get_docs(path: str):
    loader = PyPDFLoader(path)
    documents = loader.load()
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return docs


def recreate_with_docs(docs):
    Qdrant.from_documents(
        docs, embeddings, 
        url=QDRANT_HOST, prefer_grpc=True, api_key=QDRANT_API_KEY,
        collection_name="my_documents",
    )


def delete_by_source(collection_name: str, source: str):
    client = get_client()
    client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.source",
                        match=models.MatchValue(value=source)
                    )
                ]
            )
        )
    )


def add_document(collection_name: str, doc: str):
    print("getting docs")
    docs = get_docs(doc)
    print("init qdrant")
    qdrant = init_qdrant(collection_name)
    print("adding docs")
    ids_list = qdrant.add_documents(documents=docs)
    print(ids_list)



def list_all_records(collection_name: str):
    client = get_client()
    (list_of_records, _) = client.scroll(collection_name=collection_name)
    return list_of_records


def run_query(collection_name: str, chat_id: str, query_text: str):
    global index
    if index is None:
        index = init_index(collection_name)

    if REDIS_CONN is None:
        return "REDIS_CONN is not set"

    history = RedisChatMessageHistory(chat_id, url=REDIS_CONN, ttl=60*60*24)
    conversational_memory = ConversationBufferWindowMemory(
        chat_memory=history,
        memory_key='chat_history',
        k=5,
        return_messages=True
    )
    llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo') # type: ignore
    agent_chain = initialize_agent(tools, llm, verbose=True, agent='chat-conversational-react-description', memory=conversational_memory) # type: ignore
    response = agent_chain.run(input=query_text)

    return str(response)
