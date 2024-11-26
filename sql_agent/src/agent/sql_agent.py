from typing import Literal, Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed import IsLastStep
from langgraph.prebuilt import ToolNode

dialect = "PostgreSQL"
top_k = 5
system_message = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

To start you should ALWAYS look at the tables in the database to see what you can query.
Do NOT skip this step.
Then you should query the schema of the most relevant tables."""

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    is_last_step: IsLastStep


class SqlAgent:
    def __init__(self, model: BaseChatModel, memory, tools):
        self.model = model
        self.memory = memory
        self.tools = tools
        self.agent = self._build_agent()

    def _wrap_model(self) -> RunnableSerializable[AgentState, AIMessage]:
        self.model = self.model.bind_tools(self.tools)
        preprocessor = RunnableLambda(
            lambda state: [SystemMessage(content=system_message)] + state["messages"],
            name="StateModifier",
        )
        return preprocessor | self.model

    async def acall_model(self, state: AgentState, config: RunnableConfig) -> AgentState:
        model_runnable = self._wrap_model()
        response = await model_runnable.ainvoke(state, config)

        if state["is_last_step"] and response.tool_calls:
            return {
                "messages": [AIMessage(id=response.id, content="Sorry, need more steps to process this request.")]
            }
        return {"messages": [response]}

    def pending_tool_calls(self, state: AgentState) -> Literal["tools", "done"]:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            raise TypeError(f"Expected AIMessage, got {type(last_message)}")
        if last_message.tool_calls:
            return "tools"
        return "done"

    def _build_agent(self) -> StateGraph:
        # Define the graph
        agent = StateGraph(AgentState)

        agent.add_node("model", self.acall_model)
        agent.add_node("tools", ToolNode(self.tools))

        agent.add_edge(START, "model")
        agent.add_edge("tools", "model")  # Always run "model" after "tools"

        agent.add_conditional_edges(
            "model", self.pending_tool_calls, {"tools": "tools", "done": END}
        )

        # Compile the agent with a checkpointer
        return agent.compile(
            checkpointer=self.memory,
        )

    def get_agent(self):
        return self.agent
