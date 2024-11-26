#### ASYNC ####
import asyncio

from client import AgentClient


async def amain() -> None:
    client = AgentClient()

    print("Chat example:")
    response = await client.ainvoke("What is the status of order number 3?")
    for message in response.messages:
        message.pretty_print()
    


asyncio.run(amain())
