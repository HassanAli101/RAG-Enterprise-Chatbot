import streamlit as st
import requests

# Chatbot page functionality
def chatbot_page():
    # Sidebar with styling and sections
    with st.sidebar:
        st.title("🌟 Employee Assistant Chatbot")
        st.markdown("Welcome! Use this chatbot to assist with your employee-related queries.")
        
        # Document upload section
        st.subheader("📂 Document Upload")
        uploaded_file = st.file_uploader("Choose a PDF document to upload:", type="pdf")
        upload_button = st.button("Upload to Database")

        # Handle file upload
        if uploaded_file and upload_button:
            files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            response = requests.post("http://localhost:8000/employee/upload", files=files)

            if response.status_code == 200:
                st.success(f"File '{uploaded_file.name}' has been successfully uploaded to the database.")
            else:
                st.error("There was an error uploading the file.")

        # Verbose toggle section
        st.subheader("🔧 Verbose Mode")
        verbose = st.checkbox("Enable Verbose Mode", value=False)

        if verbose is not None:
            response = requests.post("http://localhost:8000/employee/changeVerbose", json={"verbose": verbose})
            if response.status_code == 200:
                st.success(f"Verbose mode toggled to: {verbose}")
            else:
                st.error("Failed to toggle verbose mode.")

        # Clear chat history section
        st.subheader("🧹 Clear Chat History")
        clear_history = st.checkbox("Clear chat history", value=False)

        if clear_history:
            st.session_state.messages = []
            response = requests.get("http://localhost:8000/employee/clearCache")
            if response.status_code == 200:
                st.success("Chat history cleared.")
            else:
                st.error("Failed to clear chat history.")

    # Function to generate response from backend
    def generate_response(input_text):
        response = requests.post("http://localhost:8000/employee", json={"query": input_text})
        if response.status_code == 200:
            return response.json().get("response", "No response available.")
        return "Sorry, I couldn't process your request."

    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your employee assistant chatbot. How can I help you today?"}
        ]

    # Chat interface
    st.subheader("💬 Chat Interface")
    chat_container = st.container()
    for message in st.session_state.messages:
        role_color = "background-color: #2e2c2c;"  # Gray background for all messages
        with chat_container.container():
            st.markdown(f"<div style='{role_color} padding: 10px; border-radius: 5px; margin: 5px 0;'>{message['content']}</div>", unsafe_allow_html=True)

    # User input and response generation
    if input := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": input})
        with chat_container:
            st.markdown(f"<div style='background-color: #2e2c2c; padding: 10px; border-radius: 5px; margin: 5px 0;'>{input}</div>", unsafe_allow_html=True)

        # Assistant's response
        with st.spinner("Working on it..."):
            response = generate_response(input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(f"<div style='background-color: #2e2c2c; padding: 10px; border-radius: 5px; margin: 5px 0;'>{response}</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    chatbot_page()