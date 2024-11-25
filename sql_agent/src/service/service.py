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

from agent import sql_agent
from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    UserInput,
)
from service.utils import langchain_to_chat_message

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Construct agent with Sqlite checkpointer
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as saver:
        sql_agent.checkpointer = saver
        app.state.agent = sql_agent
        yield
    # context manager will clean up the AsyncSqliteSaver on exit


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
