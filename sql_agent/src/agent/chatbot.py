from typing import Literal, Annotated, Callable
from typing_extensions import TypedDict

from langchain.schema.runnable import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.tools.base import BaseTool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed import IsLastStep
from langgraph.prebuilt import ToolNode

from agent.tools import ToRagAgent, ToRepresentative, ToSqlAgent

# Prompts
coordinator_prompt = """You are the Coordinator in a multi-agent chatbot system
for Uber Eats. Your role is to analyze user queries and delegate tasks to the
relevant specialized agents based on their capabilities:

1.	SQL Agent: Fetches data from the company's SQL database about Customers,
Orders, Order Items, Restaurants, Delivery Personnel, Payments,
Feedback/Complaints, and Food Items.

2.	RAG Agent: Processes pre-retrieved context containing company policies,
rules, regulations, FAQs, and customer support-related documents to extract
relevant information.

3.	Representative: Synthesizes input from agents and delivers professional,
user-friendly responses.

You have access to three tools: ToSQL, ToRAG, and ToRepresentative. Always use
one of these tools to communicate with the agents. Never communicate directly
with the user.  Evaluate the user query to decide which agent(s) to involve,
integrate their outputs, and forward the consolidated information to the
Customer Service Representative for final interaction. Ensure accuracy,
efficiency, and appropriate task allocation."""

representative_prompt = """You are an expert customer support specialist in a
multi-agent chatbot system for Uber Eats. Your role is to professionally respond
to user queries by synthesizing input from other agents and delivering accurate,
clear, and helpful responses. You are the only agent in the system that
interacts directly with the user so carefully reflect on the previous messages
to understand the context other agents have provided; no other agents can
communicate with the user.  Maintain a friendly and professional tone at all
times. Ensure that responses are customer-focused and do not disclose internal
system details or processes."""

sql_prompt_template = SystemMessagePromptTemplate.from_template(
"""You are an agent designed to interact with a SQL database.  Given an input
question, create a syntactically correct {dialect} query to run, then look at
the results of the query and return the answer.  Always limit your query to at
most {top_k} results.  You can order the results by a relevant column to return
the most interesting examples in the database.  Never query for all the columns
from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.  NOTE: THE USER
CAN'T SEE THE TOOL RESPONSE.  Only use the below tools. Only use the information
returned by the below tools to construct your final answer.  You MUST double
check your query before executing it. If you get an error while executing a
query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query.  Do NOT skip this step. Then you should query the schema of the most
relevant tables."""
)
sql_prompt = sql_prompt_template.format(
    dialect="Postgres", top_k=5
)

# Graph
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    representative_memory: Annotated[list[AnyMessage], add_messages] # selective conversation history for the assistant
    is_last_step: IsLastStep
    pending_tool_calls: list[BaseTool]
    agent_responses: list[str] # TODO: is it needed now that i've added a separate memory for the representative?

class Chatbot:
    def __init__(self, llm, rag_pipeline, sql_tools, memory):
        self.llm = llm
        self.sql_llm = llm.bind_tools(list(sql_tools.values()))
        self.coordinator_llm = llm.bind_tools([ToSqlAgent, ToRagAgent, ToRepresentative], tool_choice="required")
        self.memory = memory
        self.tools = sql_tools
        self.rag_pipeline = rag_pipeline
        self.coordinator_prompt = coordinator_prompt
        self.sql_prompt = sql_prompt
        self.representative_prompt = representative_prompt

    def _coordinator_node(self, state: State):
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage):
            messages = [{"role": "system", "content": self.coordinator_prompt}] + state["messages"]
            response = self.coordinator_llm.invoke(messages)
            # coordinator will alway communicate via tool calling
            tool_calls=response.tool_calls
            if len(tool_calls) == 1:
                return {"messages": [response], "representative_memory": [last_message]}
            response.tool_calls=[tool_calls[0]]
            return {"messages": [response], "representative_memory": [last_message], "pending_tool_calls": tool_calls[1:]}
        pending_tool_calls = state["pending_tool_calls"]
        agent_responses = state["agent_responses"]
        agent_responses.append(last_message.content)
        if pending_tool_calls:
            tool_call = pending_tool_calls.pop(0)
            response = AIMessage(content=tool_call["args"]["query"], tool_calls=[tool_call])
            return {"messages": [response], "pending_tool_calls": pending_tool_calls, "agent_responses": agent_responses}
        else:
            # TODO: can coordinator act as a representative?
            # messages = [{"role": "system", "content": representative_prompt}] + state["representative_memory"]
            messages = [{"role": "system", "content": self.coordinator_prompt}] + state["messages"]
            response = self.coordinator_llm.invoke(messages)
            return {"messages": [response]}
        # raise TypeError(f"Expected AIMessage or HumanMessage, got {type(last_message)}")

    def _customer_rag_node(self, state: State):
        query=state["messages"][-2].tool_calls[0]["args"]["query"]
        return {"messages": [self.rag_pipeline.generate(query)]}

    def _sql_agent_node(self, state: State):
        messages = [self.sql_prompt] + state["messages"]
        response = self.sql_llm.invoke(messages)
        return {"messages": [response]}

    def _representative_node(self, state: State):
        messages = [{"role": "system", "content": self.representative_prompt}] + state["representative_memory"]
        response = self.llm.invoke(messages)
        return {"messages": [response], "representative_memory": [response], "pending_tool_calls": [], "agent_responses": []} # flush the tool calls and agent rsponses

    def _route_coordinator(self, state: State):
        tool_calls = state["messages"][-1].tool_calls
        if tool_calls:
            if tool_calls[0]["name"] == ToSqlAgent.__name__:
                return "router_sql_agent"
            elif tool_calls[0]["name"] == ToRagAgent.__name__:
                return "router_rag_agent"
            elif tool_calls[0]["name"] == ToRepresentative.__name__: 
                return "to_representative"
        raise ValueError("Invalid route")

    def _create_entry_node(self, assistant_name: str, flow: Literal["one_way", "two_way"]) -> Callable:
        def entry_node(state: State) -> dict:
            tool_call_id = state["messages"][-1].tool_calls[0]["id"]
            response= {
                "messages": [
                    ToolMessage(
                        content=f"Passing dialog control to the {assistant_name}.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
            return response
        def entry_exit_node(state: State) -> dict:
            last_message = state["messages"][-1]
            # if not isinstance(last_message, AIMessage):
            #     raise TypeError(f"Expected AIMessage, got {type(last_message)}")
            if last_message.tool_calls:
                tool_call_id = last_message.tool_calls[0]["id"]
                response = {
                    "messages": [
                        ToolMessage(
                            content=f"Passing dialog control to the {assistant_name}.",
                            tool_call_id=tool_call_id,
                        )
                    ],
                    "representative_memory": [
                        AIMessage(
                            content=f"Query for {assistant_name}: {last_message.tool_calls[0]['args']['query']}",
                        )
                    ],
                }
            else: # add final output from the agent to the assistant memory
                last_message.content = f"Response from {assistant_name}: {last_message.content}"
                response = {
                    "messages": [last_message],
                    "representative_memory": [last_message],
                }
            return response
        if flow == "one_way":
            return entry_node
        elif flow == "two_way":
            return entry_exit_node

    def _handle_tool_error(self, state) -> dict:
        error = state.get("error")
        tool_calls = state["messages"][-1].tool_calls
        return {
            "messages": [
                ToolMessage(
                    content=f"Error: {repr(error)}\n please fix your mistakes.",
                    tool_call_id=tc["id"],
                )
                for tc in tool_calls
            ]
        }

    def _create_tool_node_with_fallback(self, tools: list) -> dict:
        return ToolNode(tools).with_fallbacks(
            [RunnableLambda(self._handle_tool_error)], exception_key="error"
        )

    def _pending_tool_calls(self, state: State) -> Literal["tools", "done"]:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            raise TypeError(f"Expected AIMessage, got {type(last_message)}")
        if last_message.tool_calls:
            return "tools"
        return "done"

    def _route_agent(self, state: State) -> Literal["agent", "coordinator"]:
        last_message = state["messages"][-1]
        if isinstance(last_message, ToolMessage):
            return "agent"
        return "coordinator"

    def build_graph(self) -> StateGraph:
        graph = StateGraph(State)

        graph.add_node("coordinator", self._coordinator_node)
        graph.add_node("representative", self._representative_node)
        graph.add_node("sql_agent", self._sql_agent_node)
        graph.add_node("rag_agent", self._customer_rag_node)
        graph.add_node("tools", self._create_tool_node_with_fallback(list(self.tools.values())))
        graph.add_node("router_sql_agent", self._create_entry_node("SQL Agent", flow="two_way"))
        graph.add_node("router_rag_agent", self._create_entry_node("Customer RAG Agent", flow="two_way"))
        graph.add_node("to_representative", self._create_entry_node("Customer Service Representative", flow="one_way"))

        graph.add_edge(START, "coordinator")
        graph.add_conditional_edges(
            "coordinator",
            self._route_coordinator,
            [
                "router_sql_agent",
                "router_rag_agent",
                "to_representative",
            ],
        )
        graph.add_conditional_edges(
            "router_sql_agent", self._route_agent, {"agent": "sql_agent", "coordinator": "coordinator"}
        )
        graph.add_conditional_edges(
            "router_rag_agent", self._route_agent, {"agent": "rag_agent", "coordinator": "coordinator"}
        )
        graph.add_conditional_edges(
            "sql_agent", self._pending_tool_calls, {"tools": "tools", "done": "router_sql_agent"}
        )
        graph.add_edge("tools", "sql_agent")
        graph.add_edge("rag_agent", "router_rag_agent")
        graph.add_edge("to_representative", "representative")
        graph.add_edge("representative", END)

        graph = graph.compile(checkpointer=self.memory)
        return graph

# from IPython.display import Image, display
# display(Image(graph.get_graph(xray=1).draw_mermaid_png()))

# initial_state = {
#     "messages": [],
#     "assistant_memory": [],
#     "pending_tool_calls": [],
#     "agent_responses": []
# }

# graph.update_state(config, initial_state)
