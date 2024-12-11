from typing import Literal, Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)

from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed import IsLastStep
from langgraph.prebuilt import ToolNode

system_message = """You will act as an expert routing agent where your job is to
carefully analyze the user question and route it to one of the specialized
agents in the team. You have access to tools for interacting with the database
that you can execute to get more context about the connected SQL database. To
start, you should ALWAYS look at the tables in the database to see what data is
available in the database. Do NOT skip this step. Then you should query the
schema of the most relevant tables if you need more context. After processing
all this context, you should decide whether the user question can be answered by
the SQL agent or not."""

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    is_last_step: IsLastStep

class RouterState(TypedDict):
    messages: Annotated[list, add_messages]
    route: Literal["database", "other"]

class Router(TypedDict):
    route: Literal["database", "other"]

class RouterAgent:
    def __init__(self, model: BaseChatModel, memory, tools, sql_agent: CompiledStateGraph):
        self.model = model
        self.memory = memory
        self.tools = tools
        self.sql_agent = sql_agent
        self.agent = self._build_agent()

    async def acall_model(self, state: AgentState, config: RunnableConfig) -> AgentState:
        model_runnable = self._wrap_model()
        response = await model_runnable.ainvoke(state, config)

        if state["is_last_step"] and response.tool_calls:
            return {
                "messages": [AIMessage(id=response.id, content="Sorry, need more steps to process this request.")]
            }
        return {"messages": [response]}

    def _wrap_model(self) -> RunnableSerializable[AgentState, AIMessage]:
        self.model = self.model.bind_tools(self.tools).with_structured_output(Router)
        preprocessor = RunnableLambda(
            lambda state: [SystemMessage(content=system_message)] + state["messages"],
            name="StateModifier",
        )
        return preprocessor | self.model

    def _router_node(self, state: RouterState):
        messages = [{"role": "system", "content": system_message}] + state["messages"]
        route = self.model.invoke(messages)
        return {"route": route["route"]}

    def _normal_llm_node(self, state: RouterState):
        system_message = "You are a helpful AI assistant. Answer the user questions."
        messages = [{"role": "system", "content": system_message}] + state["messages"]
        response = self.model.invoke(messages)
        return {"messages": [response]}

    def _route_after_prediction(self, state: RouterState,) -> Literal["sql_agent", "general_assistant", "tools"]:
        if state["route"] == "database":
            return "sql_agent"
        elif state["route"] == "tool_call":
            return "tools"
        else:
            return "general_assistant"

    def _build_agent(self) -> StateGraph:
        graph = StateGraph(RouterState)

        graph.add_node("coordinator", self._router_node)
        graph.add_node("general_assistant", self._normal_llm_node)
        graph.add_node("sql_agent", self.sql_agent)
        graph.add_node("tools", ToolNode(self.tools))

        graph.add_edge(START, "coordinator")
        graph.add_edge("coordinator", END)
        graph.add_conditional_edges("coordinator", self._route_after_prediction)
        graph.add_edge("tools", "coordinator")
        graph.add_edge("general_assistant", "coordinator")
        graph.add_edge("sql_agent", "coordinator")

        return graph.compile(
            checkpointer=self.memory,
        )

    def get_agent(self):
        return self.agent

