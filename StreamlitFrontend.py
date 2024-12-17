import streamlit as st
import requests

# Streamlit page configuration
st.set_page_config(page_title="Employee Assistant Chatbot", page_icon="🤖", layout="wide")

# Sidebar styling and title
with st.sidebar:
    st.title('🌟 Employee Assistant Chatbot')
    st.markdown("Welcome! Use this chatbot to assist with your employee-related queries.")
    
    # Document upload section in the sidebar
    st.subheader("📂 Document Upload")
    uploaded_file = st.file_uploader("Choose a PDF document to upload:", type="pdf")
    upload_button = st.button("Upload to Database")

    # Handle file upload
    if uploaded_file and upload_button:
        # Send the file to the FastAPI server
        files = {'file': (uploaded_file.name, uploaded_file, 'application/pdf')}
        response = requests.post("http://localhost:8000/employee/upload", files=files)

        if response.status_code == 200:
            st.success(f"File '{uploaded_file.name}' has been successfully uploaded to the database.")
        else:
            st.error("There was an error uploading the file.")

    # Verbose toggle button
    st.subheader("🔧 Verbose Mode")
    verbose = st.checkbox("Enable Verbose Mode", value=False)

    # Send the verbose state to FastAPI server as a POST request
    if verbose is not None:
        response = requests.post("http://localhost:8000/employee/changeVerbose", json={"verbose": verbose})
        if response.status_code == 200:
            st.success(f"Verbose mode toggled to: {verbose}")
        else:
            st.error("Failed to toggle verbose mode.")

    # Clear chat history checkbox
    st.subheader("🧹 Clear Chat History")
    clear_history = st.checkbox("Clear chat history", value=False)

    if clear_history:
        # Reset the messages in Streamlit session state
        st.session_state.messages = []
        # Send a request to the backend to clear the bot cache
        requests.get("http://localhost:8000/employee/clearCache")  # Trigger cache clear

# Function for generating LLM response
def generate_response(input):
    response = requests.post("http://localhost:8000/employee", json={"query": input})
    if response.status_code == 200:
        return response.json().get("response")
    return "Sorry, I couldn't process your request."

# Initial message setup
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your employee assistant chatbot. How can I help you today?"}]

# Display chat messages in a more organized style
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
