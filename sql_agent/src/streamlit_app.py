import asyncio
import os
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from collections.abc import AsyncGenerator
from client import AgentClient
from schema import ChatHistory, ChatMessage

# A Streamlit app for interacting with the langgraph agent via a simple chat interface.
# The app has three main functions which are all run async:

# - main() - sets up the streamlit app and high level structure
# - draw_messages() - draws a set of chat messages

# The app heavily uses AgentClient to interact with the agent's FastAPI endpoints.


APP_TITLE = "CS6303: Foodpanda Chatbot"
APP_ICON = "🐼"


async def main() -> None:

    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
    )

    # Hide the streamlit upper-right chrome
    st.html(
        """
        <style>
        [data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
        </style>
        """,
    )
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    if "agent_client" not in st.session_state:
        agent_url = os.getenv("AGENT_URL", "http://localhost")
        st.session_state.agent_client = AgentClient(agent_url)

    agent_client: AgentClient = st.session_state.agent_client

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = get_script_run_ctx().session_id
            # TODO: initialize graph state
            agent_client.initialize_state(thread_id)
            messages = []
        else:
            history: ChatHistory = agent_client.get_history(thread_id=thread_id)
            messages = history.messages
        st.session_state.thread_id = thread_id
        st.session_state.messages = messages

    with st.sidebar:
        st.header(f"{APP_ICON} {APP_TITLE}")
        ""
        "RAG-Powered Intelligent Chatbot for Enhanced Internal Queries and Customer Support Leveraging LangGraph, FastAPI and Streamlit"
        show_tool_calls = st.toggle("Show tool calls", value=True)

        st.markdown(
            f"Thread ID: **{st.session_state.thread_id}**",
            help=f"Set URL query parameter ?thread_id={st.session_state.thread_id} to continue this conversation",
        )

    # Draw existing messages
    messages: list[ChatMessage] = st.session_state.messages

    if len(messages) == 0:
        WELCOME = "Hello! I'm an AI-powered customer support chatbot. I may take a few seconds to boot up when you send your first message. Ask me anything!"
        with st.chat_message("ai"):
            st.write(WELCOME)

    # draw_messages() expects an async iterator over messages
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter(), show_tool_calls=show_tool_calls)

    # Generate new message if the user provided new input
    if user_input := st.chat_input():
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)
        response = await agent_client.ainvoke(message=user_input, thread_id=st.session_state.thread_id)

        for msg in response.messages:
            messages.append(msg)

        # draw new messages
        async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
            for m in response.messages:
                yield m

        await draw_messages(amessage_iter(), show_tool_calls=show_tool_calls)
        st.rerun()  # Clear stale containers

async def draw_messages(messages_agen: AsyncGenerator[ChatMessage | str, None], show_tool_calls: bool = True) -> None:
    """
    Draws a set of chat messages

    This function has additional logic to handle tool calls.
    - Use a status container to render tool calls. Track the tool inputs and outputs
        and update the status container accordingly.

    The function also needs to track the last message container in session state
    since later messages can draw to the same container.

    Args:
        messages_aiter: An async iterator over messages to draw.
        show_tool_call: Whether to render the tool call containers or not.
    """

    # Keep track of the last message container
    last_message_type = None
    st.session_state.last_message = None

    # Iterate over the messages and draw them
    while msg := await anext(messages_agen, None):
        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()
        match msg.type:
            # A message from the user, the easiest case
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            # A message from the agent is the most complex case, since we need to
            # handle tool calls.
            case "ai":
                # If the last message type was not AI, create a new chat message
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

                with st.session_state.last_message:
                    # If the message has content, write it out.
                    if msg.content:
                        st.write(msg.content)

                    if msg.tool_calls:
                        # Create a status container for each tool call and store the
                        # status container by ID to ensure results are mapped to the
                        # correct status container.
                        call_results = {}
                        for tool_call in msg.tool_calls:
                            if show_tool_calls:
                                status = st.status(
                                    f"""Tool Call: {tool_call["name"]}""",
                                    state="complete",
                                )
                                call_results[tool_call["id"]] = status
                                status.write("Input:")
                                status.write(tool_call["args"])
                            else:
                                call_results[tool_call["id"]] = -1

                        # Expect one ToolMessage for each tool call.
                        for _ in range(len(call_results)):
                            tool_result: ChatMessage = await anext(messages_agen)
                            if tool_result.type != "tool":
                                st.error(f"Unexpected ChatMessage type: {tool_result.type}")
                                st.write(tool_result)
                                st.stop()

                            # Record the message if it's new, and update the correct
                            # status container with the result
                            if show_tool_calls:
                                status = call_results[tool_result.tool_call_id]
                                status.write("Output:")
                                status.write(tool_result.content)
                                status.update(state="complete")

            # In case of an unexpected message type, log an error and stop
            case _:
                st.error(f"Unexpected ChatMessage type: {msg.type}")
                st.write(msg)
                st.stop()

if __name__ == "__main__":
    asyncio.run(main())
