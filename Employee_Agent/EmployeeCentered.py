#pip installing:
# %pip install langchain
# %pip install langchain_community
# %pip install langchain_huggingface
# %pip install langchain_pinecone
# %pip install pinecone
# %pip install pinecone-client
# %pip install dotenv
# %pip install streamlit
# %pip install pymupdf
# %pip install -qU langchain_community wikipedia
# %pip install --upgrade --quiet langchain-text-splitters tiktoken
# %pip install difflib

import os
import langchain #its giving module not found error
import langchain_community
import langchain_huggingface
import langchain_pinecone
import pinecone
import dotenv
import streamlit as st

# Additional Imports (loading document):
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter

#pinecone etc (storage of ducments):
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from uuid import uuid4

#hugging face etc (for generation):
from langchain_huggingface import HuggingFaceEndpoint
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnableLambda

#memory imports
#I used these documentations: https://python.langchain.com/v0.1/docs/use_cases/chatbots/memory_management/ , https://python.langchain.com/v0.1/docs/modules/memory/types/buffer/ , https://python.langchain.com/v0.1/docs/modules/memory/
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.chains import create_history_aware_retriever #new
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

#caching imports:
from difflib import SequenceMatcher

from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import TokenTextSplitter
#for timing the retrivals
import time

#for parsing:
import re

dotenv.load_dotenv()

class EmployeeChatBot:
    # TODO: To be implemented
    def __init__(self):
        #loading variables:
        self.combined_text = ""
        self.CHUNK_SIZE = 256
        self.CHUNK_OVERLAP = 0.50
        #storing variables:
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self.index_name = "employee-queries-db" #keep the name small
        self.embeddings = HuggingFaceEmbeddings()
        self.index = self.pc.Index(self.index_name) #Remember, i can do this because i have already once created this index, else create index first
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embeddings)
        # generating variables
        self.retriever = self.vector_store.as_retriever( search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.5},) #tunable
        self.repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1" #tunable
        self.llm = HuggingFaceEndpoint( repo_id=self.repo_id, temperature= 0.8, top_k= 50, huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY') ) #tunable

        #memory variables:
        self.chat_history = []
        self.system_instruction = """Given a chat history and the latest user question \
            which might reference context in the chat history, formulate a standalone question \
            which can be understood without the chat history. Do NOT answer the question, \
            just reformulate it if needed and otherwise return it as is.""" #key observation

        self.memory_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_instruction),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")])

        self.history_aware_retriever = create_history_aware_retriever(
            self.llm,
            self.retriever,
            self.memory_prompt
        )

        self.system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        self.document_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)

        self.retrieval_chain = create_retrieval_chain(self.history_aware_retriever, self.document_chain)

        self.store = {}

        self.conversational_rag_chain = RunnableWithMessageHistory(
            self.retrieval_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        #cache variables
        self.cache = []

    # TODO: To be implemented
    def generate(self, query):
      # query_response = self.full_chain.invoke({"question": query})
      # Check the cache for a similar query
      for cached_query, cached_response in self.cache:
          if self.similar(cached_query, query) > 0.6:
              print("cache found, with: ", cached_query, " score: ", self.similar(cached_query, query))
              return cached_response


      query_response = self.conversational_rag_chain.invoke(
          {"input": query},
          config={
              "configurable": {"session_id": "abc123"}
          },  # constructs a key "abc123" in `store`.
      )["answer"]

      self.cache.append([query, query_response])
      # print("query response is: ", query_response)

      return query_response

    def similar(self, a, b):
        return SequenceMatcher(None, a, b).ratio()

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]



    #this function will add the given filepath (as a string) to the pinecone vector db after parsing it
    def AddFileToDB(self, docs_to_load):
      # [ADD LOADING AND PARSING AND CHUNKING PART HERE]
      combined_text = ""
      for doc in docs_to_load:
        loader = PyMuPDFLoader(doc)
        documents = loader.load()
        # print(documents)
        for page in documents:
          text = page.page_content
          if "contents" in text.lower():
            continue
          text = re.sub(r'\bPage\s+\d+\b', '', text, flags=re.IGNORECASE)
          text = re.sub(r'\n', '', text).strip() #removing all newlines
          # print(text)
          text = re.sub(r'[^\w\s.,?!:;\'\"()&-]', '', text)
          combined_text += text + " "
      combined_text = combined_text.strip()
      # print(combined_text)
      text_splitter = TokenTextSplitter(chunk_size=self.CHUNK_SIZE, chunk_overlap=int(self.CHUNK_SIZE*self.CHUNK_OVERLAP))
      texts = text_splitter.split_text(combined_text)
      docs = text_splitter.create_documents(texts)
      # print(docs)
      if self.index_name not in self.pc.list_indexes().names():
        self.pc.create_index(  #tunable
          name=self.index_name,
          dimension=768,
          metric="cosine",
          spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
          )
        )
      embeddings = HuggingFaceEmbeddings()
      index = self.pc.Index(self.index_name)
      vector_store = PineconeVectorStore(index=index, embedding=embeddings)
      uuids = [str(uuid4()) for _ in range(len(docs))]
      vector_store.add_documents(documents=docs, ids=uuids)



# Initialize the EmployeeChatBot
bot = EmployeeChatBot()


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
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Pass the temporary file path to AddFileToDB
        bot.AddFileToDB([temp_file_path])
        st.success(f"File '{uploaded_file.name}' has been successfully uploaded to the database.")

# Function for generating LLM response
def generate_response(input):
    result = bot.generate(input)
    return result

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