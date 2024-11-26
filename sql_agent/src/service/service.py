import os
import logging
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, FastAPI, HTTPException
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

from agent import SqlAgent

from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    UserInput,
)
from service.utils import langchain_to_chat_message

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

def init_db():
    db_ro = SQLDatabase.from_uri(os.getenv("DATABASE_URI_RO"), engine_args={"pool_pre_ping": True})
    db_rw = SQLDatabase.from_uri(os.getenv("DATABASE_URI_RW"), engine_args={"pool_pre_ping": True})
    return {"db_ro": db_ro, "db_rw": db_rw}

def init_db_toolkit(db: SQLDatabase):
    db_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    db_tools={tool.name:tool for tool in db_toolkit.get_tools()}
    return db_tools

def init_sql_agent(memory, db: SQLDatabase, tools: list):
    sql_agent = SqlAgent(model=llm, memory=memory, db=db, tools=tools)
    return sql_agent.get_agent()

@asynccontextmanager
async def checkpointer() -> AsyncGenerator[None, None]:
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as memory:
        yield memory

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    db_connections = init_db()
    db_tools = init_db_toolkit(db_connections["db_ro"])
    db_ro_tools = [db_tools.get(key) for key in ["sql_db_list_tables", "sql_db_schema"]]
    db_tools = init_db_toolkit(db_connections["db_rw"])
    db_rw_tools = list(db_tools.values())

    async with checkpointer() as memory:
        sql_agent_ro = init_sql_agent(memory, db_connections["db_ro"], db_ro_tools)
        app.state.agent = sql_agent_ro
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


app.include_router(router)
