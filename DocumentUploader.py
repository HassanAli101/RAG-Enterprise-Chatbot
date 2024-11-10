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

import os
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
import tempfile

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
        # self.memory_template = """You are a ambiguity clearer, your task is to examine the human question and check for any "he/she/it/they/them" ambiguities.
        # return an updated human question fixing those ambiguities using the previous conversation context only.
        # if there is not enought relevant context, RETURN HUMAN QUESTION AS IT IS
        # YOUR ANSWER SHOULD BE A QUESTION WHICH ONLY CLARIFIES ANY AMBIGUITY IN human question by replacing it with their name
        # RETURN IN FORMAT: New human question: (updated question)
        # Previous conversation:
        # {chat_history}

        # human question: {question}
        # New human question:
        # """
        # self.memory_prompt = PromptTemplate.from_template(self.memory_template)

        # self.memory = ConversationBufferMemory(memory_key="chat_history")
        # self.conversation = LLMChain(
        #     llm=self.llm,
        #     prompt=self.memory_prompt,
        #     verbose=False, #okay.... so this setting tells everything going on in the computations and stuff
        #     memory=self.memory
        # )

        # #prompt variables
        # self.Classifier_template = """
        # You are a prompt classifier designed to classify questions from employees in an organization.
        # classify the following question into "Relevant" or "Irrelevant", based on whether the query theme is of a question from an organization employee, the question could be about IT, HR, Finance or any other department
        # Only answer from the specified classes and one word answers.

        # Question: {question}
        # Answer:
        # """
        # self.Classifier_prompt = PromptTemplate( template=self.Classifier_template, input_variables=["question"] )

        # #chain variables
        # self.classifier_chain = ({"question": RunnablePassthrough()} | self.Classifier_prompt | self.llm  | StrOutputParser() )


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
      print(docs)
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


    # TODO: To be implemented
    def generate(self, query):
        return "THIS IS A BOT GENERATED RESPONSE"


    #Helper functions:
    def format_docs(self, docs):
        return "\n\n".join([d.page_content for d in docs])


    def route(self, info):
        pass

# Initialize the EmployeeChatBot
bot = EmployeeChatBot()

# Streamlit page configuration
st.set_page_config(page_title="Employee Assistant Chatbot")
with st.sidebar:
    st.title('Employee Assistant Chatbot')

# Function for generating LLM response
def generate_response(input):
    result = bot.generate(input)
    return result

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am an employee assistant chatbot. How can I help you today?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Working on it.."):
            response = generate_response(input) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)

# File upload section
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
if uploaded_file is not None:
    # Save the uploaded file to a temporary file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Pass the temporary file path to AddFileToDB
    bot.AddFileToDB([temp_file_path])
    st.success(f"File '{uploaded_file.name}' has been added to the database.")

