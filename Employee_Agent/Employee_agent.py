from langchain.agents import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from EmployeeCentered import EmployeeChatBot

# Initialize your chatbot
bot = EmployeeChatBot()

# Define the tools for LangChain
class ChatbotTool(Tool):
    def __init__(self):
        super().__init__(
            name="Chatbot",
            func=self.run,  # Link the tool's main function
            description="Use this tool to interact with the employee assistant chatbot."
        )

    def run(self, query: str):
        # Define the logic for the Chatbot tool
        response = bot.generate(query)
        return response


class UploadFileTool(Tool):
    def __init__(self):
        super().__init__(
            name="Upload File",
            func=self.run,  # Link the tool's main function
            description="Use this tool to upload files to the database."
        )

    def run(self, file_path: str):
        # Define the logic for the Upload File tool
        bot.AddFileToDB([file_path])
        return f"File {file_path} uploaded successfully."

# Initialize LangChain Tools
chatbot_tool = ChatbotTool()
upload_file_tool = UploadFileTool()

# Define the tools in a list
tools = [chatbot_tool, upload_file_tool]

# Create a prompt for the agent

prompt_template = """
You are an assistant for employee-related queries. You can:
- Answer questions related to employee information using the tools provided.
- Upload documents to the database.

Follow this process:
1. Analyze the input.
2. Decide which tool to use.
3. Use the tool and summarize the results.

Intermediate steps will be stored in the agent's scratchpad.

Input: {input}
Agent Scratchpad: {agent_scratchpad}
Output: """
prompt = PromptTemplate(input_variables=["input", "agent_scratchpad"], template=prompt_template)

agent = create_tool_calling_agent(bot.llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_scratchpad = ""  # Start with an empty scratchpa
print(agent_executor.invoke({"input": "Upload the document", "agent_scratchpad": agent_scratchpad}))
