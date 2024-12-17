import httpx

from customer_interface.schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    UserInput,
)

class AgentClient:
    """Client for interacting with the agent service.  """
    def __init__(self, base_url: str = "http://localhost:80", timeout: float | None = None) -> None:
        """
        Initialize the client.

        Args:
            base_url (str): The base URL of the agent service.
        """
        self.base_url = base_url
        self.timeout = timeout

    async def ainvoke(self, message: str, thread_id: str | None = None) -> list[ChatMessage]:
        """
        Invoke the agent asynchronously. Only the final message is returned.

        Args:
            message (str): The message to send to the agent
            thread_id (str, optional): Thread ID for continuing a conversation

        Returns:
            AnyMessage: The response from the agent
        """
        request = UserInput(message=message)
        if thread_id:
            request.thread_id = thread_id
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{self.base_url}/invoke", json=request.model_dump(), timeout=self.timeout)
            if response.status_code == 200:
                response_object = response.json()
                return ChatHistory.model_validate(response_object)
            else:
                raise Exception(f"Error: {response.status_code} - {response.text}")

    def get_history(self, thread_id: str,) -> ChatHistory:
        """
        Get chat history.

        Args:
            thread_id (str, optional): Thread ID for identifying a conversation
        """
        request = ChatHistoryInput(thread_id=thread_id)
        response = httpx.post(f"{self.base_url}/history", json=request.model_dump(), timeout=self.timeout)
        if response.status_code == 200:
            response_object = response.json()
            return ChatHistory.model_validate(response_object)
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")

    def initialize_state(self, thread_id: str):
        """
        Initialize the state graph with empty memory.

        Args:
            thread_id (str, optional): Thread ID for identifying a conversation
        """
        request = ChatHistoryInput(thread_id=thread_id)
        response = httpx.post(f"{self.base_url}/init", json=request.model_dump(), timeout=self.timeout)
        if response.status_code == 200:
            pass
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")
