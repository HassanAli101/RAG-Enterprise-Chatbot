import os
import asyncio
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from pinecone import Pinecone
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint

from agent import SqlAgent, RouterAgent
from agent.customer_rag_agent import DocumentLoader, VectorStore, CustomerRagAgent

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

def init_db():
    db_ro = SQLDatabase.from_uri(os.getenv("DATABASE_URI_RO"), engine_args={"pool_pre_ping": True})
    db_rw = SQLDatabase.from_uri(os.getenv("DATABASE_URI_RW"), engine_args={"pool_pre_ping": True})
    return {"db_ro": db_ro, "db_rw": db_rw}

def init_db_toolkit(db: SQLDatabase):
    db_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    db_tools={tool.name:tool for tool in db_toolkit.get_tools()}
    return db_tools

def init_sql_agent(memory, tools: list):
    sql_agent = SqlAgent(model=llm, memory=memory, tools=tools)
    return sql_agent.get_agent()

async def stream_graph_updates(graph, user_input: str, config):
    messages = await graph.ainvoke(
        {"messages": [HumanMessage(content=user_input)]}, config, stream_mode="values"
    )
    for message in messages["messages"]:
        message.pretty_print()

async def arun_sql_agent():
    # query="what is the status of order number 3?"
    query="How tall is Burj Khalifa?"
    db_connections = init_db()
    db_tools = init_db_toolkit(db_connections["db_ro"])
    db_ro_tools = [db_tools.get(key) for key in ["sql_db_list_tables", "sql_db_schema"]]
    db_tools = init_db_toolkit(db_connections["db_rw"])
    db_rw_tools = list(db_tools.values())

    sql_agent_ro = init_sql_agent(MemorySaver(), db_ro_tools)

    config = {"configurable": {"thread_id": "1"}}

    await stream_graph_updates(sql_agent_ro, query, config)
    router = RouterAgent(model=llm, memory=MemorySaver(), tools=db_ro_tools, sql_agent=sql_agent_ro)
    router_agent=router.get_agent()

    graph_png = router_agent.get_graph(xray=1).draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(graph_png)

def run_customer_rag_agent():
    # Initialize document loader
    doc_loader = DocumentLoader()

    # Initialize vector store
    pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    vector_store = VectorStore(
        pinecone_client=pinecone_client,
        index_name="customer-queries-db",
        embedding_model=HuggingFaceEmbeddings(),
    )

    # Initialize LLM and memory
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.8,
        top_k=50,
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
    )
    memory = ConversationBufferMemory(memory_key="chat_history")

    # Create orchestrator
    rag_pipeline = CustomerRagAgent(
        document_loader=doc_loader,
        vector_store=vector_store,
        llm=llm,
        memory=memory,
    )

    # response = rag_pipeline.generate("What is the organization's policy on refunds?")
    # print(response)
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            response=rag_pipeline.generate(user_input)
            print(response)
        except:
            print("Invalid Input!")
            break

if __name__ == "__main__":
    run_customer_rag_agent()
    # asyncio.run(arun_sql_agent())