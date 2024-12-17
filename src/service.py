import os
import logging
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, FastAPI, HTTPException, UploadFile
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

from employee_interface.EmployeeCentered import EmployeeChatBot
from pydantic import BaseModel

from customer_interface.agent import CustomerRagAgent, DocumentLoader, VectorStore, Chatbot

from customer_interface.schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    UserInput,
)
from utils import langchain_to_chat_message

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

def init_sql_tools(permission):
    if permission == "read_only":
        db = SQLDatabase.from_uri(os.getenv("DATABASE_URI_RO"), engine_args={"pool_pre_ping": True})
    elif permission == "read_write":
        db = SQLDatabase.from_uri(os.getenv("DATABASE_URI_RW"), engine_args={"pool_pre_ping": True})
    else:
        raise ValueError("Invalid permission")
    db_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    db_tools={tool.name:tool for tool in db_toolkit.get_tools()}
    return db_tools

def init_customer_rag_pipeline():
    doc_loader = DocumentLoader()
    pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    vector_store = VectorStore(
        pinecone_client=pinecone_client,
        index_name="customer-queries-db",
        embedding_model=HuggingFaceEmbeddings(),
    )
    rag_pipeline = CustomerRagAgent(
        document_loader=doc_loader,
        vector_store=vector_store,
        llm=llm,
    )
    return rag_pipeline

@asynccontextmanager
async def checkpointer() -> AsyncGenerator[None, None]:
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as memory:
        yield memory

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    db_tools = init_sql_tools("read_only")
    rag_pipeline = init_customer_rag_pipeline()

    async with checkpointer() as memory:
        customer_chatbot = Chatbot(llm, rag_pipeline, db_tools, memory)
        app.state.agent = customer_chatbot.build_graph()
        yield

app = FastAPI(lifespan=lifespan)
router = APIRouter()


def _parse_input(user_input: UserInput) -> tuple[dict[str, Any]]:
    thread_id = user_input.thread_id or str(uuid4())
    kwargs = {
        "input": {"messages": [HumanMessage(content=user_input.message)]},
        "config": RunnableConfig(
            configurable={"thread_id": thread_id},
        ),
    }
    return kwargs


@router.post("/invoke")
async def invoke(user_input: UserInput) -> ChatHistory:
    """
    Invoke the agent with user input to retrieve a final response.

    Use thread_id to persist and continue a multi-turn conversation.
    """
    agent: CompiledStateGraph = app.state.agent
    kwargs = _parse_input(user_input)
    try:
        response = await agent.ainvoke(**kwargs, stream_mode="updates")
        output: list[ChatMessage] = [
            langchain_to_chat_message(message) for event in response for message in next(iter(event.values()))['messages']
        ]
        return ChatHistory(messages=output)
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


@router.post("/history")
def history(input: ChatHistoryInput) -> ChatHistory:
    """
    Get chat history.
    """
    agent: CompiledStateGraph = app.state.agent
    try:
        state_snapshot = agent.get_state(
            config=RunnableConfig(
                configurable={
                    "thread_id": input.thread_id,
                }
            )
        )
        messages: list[AnyMessage] = state_snapshot.values["messages"]
        chat_messages: list[ChatMessage] = [
            langchain_to_chat_message(m) for m in messages
        ]
        return ChatHistory(messages=chat_messages)
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


@router.post("/init")
def init(input: ChatHistoryInput):
    """
    Initialize State Graph with empty memory.
    """
    agent: CompiledStateGraph = app.state.agent
    try:
        agent.update_state(
            RunnableConfig(
                configurable={
                    "thread_id": input.thread_id,
                }
            ),
            {
                "messages": [],
                "assistant_memory": [],
                "pending_tool_calls": [],
                "agent_responses": []
            }
        )
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")

employee_chatbot = EmployeeChatBot()

class QueryRequest(BaseModel):
    query: str

class VerboseRequest(BaseModel):
    verbose: bool  # We expect a boolean value for the verbose flag

@app.post("/employee")
async def employee_query(request: QueryRequest):
    print("the query is: ", request.query)  # Access the query from the request object
    response = employee_chatbot.generate(request.query)
    return {"response": response}

@app.post("/employee/upload")
async def upload_document(file: UploadFile):
    temp_file_path = f"/tmp/{file.filename}"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(await file.read())
    employee_chatbot.AddFileToDB([temp_file_path])
    return {"message": f"File {file.filename} uploaded successfully"}

@app.post("/employee/changeVerbose")
def change_verbose(request: VerboseRequest):
    employee_chatbot.verbose = request.verbose  # Set the verbose flag to the value sent in the request
    return {"verbose": employee_chatbot.verbose}

@app.get("/employee/clearCache")
def clear_cache():
    employee_chatbot.cache = []  # Clears the employee_chatbot's cache
    return {"message": "Cache cleared successfully"}

app.include_router(router)
