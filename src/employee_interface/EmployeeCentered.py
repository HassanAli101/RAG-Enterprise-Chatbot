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
# %pip install cohere

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

#for cohere:
import cohere

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
        self.repo_id = "mistralai/Mistral-7B-Instruct-v0.3" #tunable
        self.llm = HuggingFaceEndpoint( repo_id=self.repo_id, temperature= 1, top_k= 50, huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY') ) #tunable
        self.verbose = False #change this to see the explanations of how the LLM reached its conclusion
        #cache variables
        self.cache = []
        self.max_cache_limit = 100
        self.Cohere_client = cohere.Client(api_key=os.environ.get("COHERE_API_KEY"))

        self.Guardrail_template = """
          You are a quality-assurance model designed to assess whether a given answer correctly and clearly responds to the question.

          Your task is to reason through the question and answer, and determine if the answer directly and accurately responds to the question, without hallucination or incorrect information. Follow these steps:

          1. **Analyze the Question**: Understand the intent of the question and what kind of answer is expected.
          2. **Evaluate the Answer**: Check if the answer directly addresses the question with factual and relevant information. Identify if there are any signs of hallucination or errors.
          3. **Assess the Clarity**: Ensure the answer is clear and concise, without unnecessary ambiguity.
          4. **Reasoning**: If there is any slight indication of hallucination or discrepancy in the answer, mark it as "Incorrect". If the answer is accurate and clearly addresses the question, mark it as "Correct".

          Use the context of this question-answer pair and evaluate accordingly.

          ### Few-Shot Examples:

          Example 1:
          QA_pair: "Question: What is the capital of France?\nAnswer: Paris"
          Chain of Thought: 
          - The question asks for the capital of France.
          - The answer "Paris" is correct and directly addresses the question.
          Answer: Correct

          Example 2:
          QA_pair: "Where can i find the user key definitions for rating?\nAnswer: The keys an not be found in connercia's documents as they are not related"
          Chain of Thought:
          - The question asks about keys for ratings.
          - The answer is incorrect because it is referring to some other documents.
          Answer: Incorrect

          ### Now evaluate the following:

          QA_pair: {question}

          Chain of Thought:
          - Analyze the question and answer.
          - Assess if the answer is accurate and directly related to the question.
          - If hallucination or incorrect information is detected, output "Incorrect".
          - If the answer is accurate and clear, output "Correct".

          Answer:
          """


        self.memory_template = """
          You are an intelligent assistant designed to enhance the clarity of questions by analyzing their context and comparing them to previously asked questions. Your task is to:

          1. Review the list of previously asked questions to understand recurring themes, topics, or context.
          2. Compare the new question to the previous ones and determine if it is related to any of them.
          3. If the new question is related, rewrite the question so that it is complete and unambiguous, using the relevant context from the previous questions.
          4. If the new question is unrelated, return it as is.

          Here are some examples:

          ### Example 1:
          Previously Asked Questions:
          1. How do I reset my email password?
          2. Where can I find the IT support portal?

          New Question: What is the process?
          Clarified Question: What is the process for resetting my email password?

          ### Example 2:
          Previously Asked Questions:
          1. How do I submit my expense report?
          2. What is the approval process for reimbursements?

          New Question: How long does it take?
          Clarified Question: How long does it take to approve an expense report?

          ### Example 3:
          Previously Asked Questions:
          1. What is the weather like today?
          2. Is it raining outside?

          New Question: What should I wear?
          Clarified Question: What should I wear based on the weather today?

          ### Example 4:
          Previously Asked Questions:
          1. How do I change my direct deposit details?

          New Question: How do I reset my password?
          Clarified Question: How do I reset my password?

          Instructions:
          - If related, provide the clarified question in a complete form.
          - If unrelated, provide the question as is.

          Now process the following input:

          {question}

          Clarified Question:
        """


        self.General_template = """
          You are a helpful assistant tasked with responding to queries that are classified as "Irrelevant" to organizational matters. When given such a query:

          1. Politely inform the user that their question does not pertain to organizational topics like IT, HR, Finance, or other departments.
          2. If possible, redirect the user to appropriate resources or provide general guidance relevant to their question.
          3. Maintain a polite, professional, and neutral tone in all responses.

          Here are some examples:

          Example 1:
          User Query: "What is the weather like today?"
          Response: "This query is unrelated to organizational topics. You can check the weather using a reliable weather app or website like Weather.com."

          Example 2:
          User Query: "What is the best way to cook pasta?"
          Response: "This query is not related to organizational matters. However, you can explore cooking tips on platforms like AllRecipes or YouTube for detailed instructions."

          Example 3:
          User Query: "How do I improve my fitness level?"
          Response: "Your query is unrelated to organizational topics, but you might find helpful fitness tips on apps like MyFitnessPal or consulting a professional trainer."

          Now respond to the following query:

          User Query: {question}
          Response:
          """


        self.Classifier_template = """
          You are a prompt classifier designed to classify questions from employees in an organization.
          Your task is to classify the following question into "Relevant" or "Irrelevant," based on whether the query theme is related to an organization employee's concerns. These could include IT, HR, Finance, or any other department.

          To determine the classification, follow these steps:
          1. Analyze the question to identify its theme or context.
          2. Determine if the question relates to organizational matters or internal operations.
          3. Classify the question as "Relevant" if it pertains to IT, HR, Finance, or any other organizational department.
          4. Classify the question as "Irrelevant" if it is unrelated to organizational matters or concerns.

          Here are examples to guide you:
          Example 1:
          Question: "How can I reset my email password?"
          Thought Process: The question is about IT support, which is an organizational concern.
          Answer: Relevant

          Example 2:
          Question: "What is the weather like today?"
          Thought Process: The question is unrelated to any organizational department.
          Answer: Irrelevant

          Example 3:
          Question: "How do I submit my expense report for reimbursement?"
          Thought Process: The question is about Finance, a department within the organization.
          Answer: Relevant

          Now classify the following question. Provide only one-word answers ("Relevant" or "Irrelevant").

          Question: {question}
          Answer:
        """
        
        self.Employee_Template = """
            You are a highly knowledgeable and reflective chatbot designed to assist employees of an organization by answering their questions accurately and thoughtfully.
            Your goal is to provide well-reasoned and clear answers based on the provided context.

            Follow these steps to construct your response:
            1. **Understand the question**: Restate the question in simpler terms if necessary, ensuring you grasp the key aspects of what is being asked.
            2. **Analyze the context**: Examine the provided context and identify relevant information that applies to the question.
            3. **Evaluate implications**: Consider any potential rules, policies, or ethical considerations that could affect the answer.
            4. **Provide the answer**: Deliver a clear, concise, and actionable response based on your analysis.
            5. **Reflection**: Briefly explain your reasoning process to ensure transparency and to help the employee understand your conclusion.

            Examples:
            ---
            Context:
            "Employees are prohibited from accepting gifts valued over $50 from clients. If a gift exceeds this amount, it must be declined or reported to the ethics committee."

            Question:
            "One of Comerica's clients is hosting an open house that includes a raffle for some free airline tickets. If I win, can I accept the tickets?"

            Answer:
            1. **Understand the question**: Can the employee accept free airline tickets won in a raffle at a client's event?
            2. **Analyze the context**: The policy prohibits accepting gifts over $50. Airline tickets are likely valued well over this limit and would need to be reported or declined.
            3. **Evaluate implications**: Accepting the tickets could violate the company's ethics policy, even if won in a raffle, as they are provided by a client.
            4. **Provide the answer**: No, you should not accept the tickets without first consulting the ethics committee to determine whether an exception applies.
            5. **Reflection**: I based my answer on the explicit policy regarding gift value limits and the need to maintain ethical boundaries with clients.
            ---
            Context:
            "Employees are allowed to attend client-sponsored events, such as dinners or conferences, provided the primary purpose is business-related and attendance has been pre-approved by their manager."

            Question:
            "A client has invited me to a dinner event to discuss our ongoing project. Do I need approval to attend?"

            Answer:
            1. **Understand the question**: Does the employee need prior approval to attend a client dinner for business purposes?
            2. **Analyze the context**: The policy states that attendance at client events requires pre-approval from the employee’s manager.
            3. **Evaluate implications**: While the event seems business-related, attending without prior approval could breach company protocol.
            4. **Provide the answer**: Yes, you need to get approval from your manager before attending the dinner.
            5. **Reflection**: My answer aligns with the policy, ensuring adherence to company guidelines while allowing participation in legitimate business activities.
            ---
            {question}
            Answer:
        """

        self.Augment_Prompt_Template = """
            The following are the file names available in our database:
            HR:
            - Code-of-conduct
            - Compensation-Benefits-Guide
            - Employee-appraisal-form
            - Employee-Handbook
            - Employee-Termination-Policy
            - Health-and-Safety-Guidelines
            - Onboarding-Manual
            - Remote-Work-Policy

            IT:
            - Cybersecurity-for-Employees
            - System-Access-Control-Policy
            - Technology-Devices-Policy

            Finance:
            - Expense-Report

            Given the following query:
            {question}

            You are tasked with identifying and returning the names of the **two most relevant files**, separated by "and," that are most helpful for addressing the query.
            do NOT provide reasoning or add any other text, just the names of files
            """
        self.Guardrail_prompt = PromptTemplate(template=self.memory_template, input_variables=["question"]) 
        self.Memory_prompt = PromptTemplate(template=self.memory_template, input_variables=["question"])
        self.General_prompt = PromptTemplate( template=self.General_template, input_variables=["question"] )
        self.Classifier_prompt = PromptTemplate( template=self.Classifier_template, input_variables=["question"] )
        self.Employee_prompt = PromptTemplate(template=self.Employee_Template, input_variables=["context", "question"] )
        self.get_relevant_docs_prompt = PromptTemplate( template=self.Augment_Prompt_Template, input_variables=["question"] )

        #chain variables
        self.guardrail_chain = ({"question": RunnablePassthrough()} | self.Guardrail_prompt | self.llm  | StrOutputParser() ) 
        self.memory_chain = ({"question": RunnablePassthrough()} | self.Memory_prompt | self.llm  | StrOutputParser() )
        self.General_chain = ({"question": RunnablePassthrough()} | self.General_prompt | self.llm  | StrOutputParser() )
        self.classifier_chain = ({"question": RunnablePassthrough()} | self.Classifier_prompt | self.llm  | StrOutputParser() )
        self.get_relevant_docs_chain = ({"question": RunnablePassthrough()} | self.get_relevant_docs_prompt | self.llm  | StrOutputParser() )
        self.Employee_chain = ({"question": RunnablePassthrough()} | self.Employee_prompt | self.llm | StrOutputParser() )


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
    #   print(docs)
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
      #Check Against Cache
        # print("[IN EMPLOYEE CENTERED], query receiveD: ", query)
        for cached_query, cached_response in self.cache:
          if self.similar(cached_query, query) > 0.6:
              print("cache found, with: ", cached_query, " score: ", self.similar(cached_query, query))
              query_response = cached_response
              if self.verbose:
                return "[VERBOSE] cache found, with previous query: " + cached_query +  " score: "+ str(self.similar(cached_query, query)) + "\nYour Answer: " + query_response
              else:
                match = re.search(r"\*\*Provide the answer\*\*: (.*?)(?:\n|$)", query_response)
                return match.group(1) if match else query_response

      #Add memory step here, implement by self, give all prev asked questions.
        if self.cache: 
          previous_questions = "\n".join([entry[0] for entry in self.cache])
        else:
          previous_questions = ""
        memory_query = "Previous_questions: \n" + previous_questions + "\nNew Question: \n" + query
        memory_response = self.memory_chain.invoke({"question": memory_query})
        if not memory_response or memory_response.strip() == "":
          memory_response = query
        query_response = "[VERBOSE] the augmented memory prompt is: " + memory_response + "\n" 
        # print("[IN EMPLOYEE CENTERED], memory response is: ", memory_response)

      #Classiify whether General question or relevant to Organization.
        classifier_response = self.classifier_chain.invoke({"question": memory_response})
        match = re.search(r'\b(Relevant|Irrelevant)\b', classifier_response)
        query_class = match.group(0) if match else None

      #Run the General Response
        if query_class == "Irrelevant":
          general_response = self.General_chain.invoke({"question": memory_response})
          self.cache.append([query, general_response])
          if self.verbose: 
            query_response += "[VERBOSE] Query classfieid as general, \nYour Answer: " + general_response.strip('"').strip()
          else:
            query_response += general_response.strip('"').strip()

      #Run the Employee Docs RAG Steps
        else:
          relevant_docs = self.get_relevant_docs(memory_response)
          search_query = memory_response + " try to answer from " + relevant_docs
          retrieved_docs = self.format_docs_rerank(self.vector_store.similarity_search(search_query))
          # print("[IN EMPLOYEE CENTERED], retrieved docs: : ", retrieved_docs)
          reranked_docs = self.rerank(memory_response, retrieved_docs)
          context = self.reformat_docs(reranked_docs)
          contextualised_query = "Context: \n" + context + "\n Question: \n" + memory_response
          if self.verbose:
            query_response += "[VERBOSE] Query classfieid as Relevant, \nYour Answer: "  + self.Employee_chain.invoke({"question": contextualised_query})
          else:
            query_response += self.Employee_chain.invoke({"question": contextualised_query})
        
      #Final GuardRailCheck if the answer actually answers the question (if needed) It is needed now, if hallucinations increase a lot, recommend to clean cache
        QA_pair = "Question: \n" + query + " \nAnswer: \n" + query_response
        guardrail_response = self.guardrail_chain.invoke({"question": QA_pair})
        match = re.search(r'\b(Correct|Incorrect)\b', guardrail_response.strip(), re.IGNORECASE)
        if match and match.group(0) != "Correct":
          if self.verbose:
            return "The Chatbot seems to be hallucinating, consider clearing the memory or calling an AI engineer to look into it since you have verbose enabled, the response was: \n" + query_response
          else:
            return "The Chatbot seems to be hallucinating, consider clearing the memory or calling an AI engineer to look into it"

      #Store in Cache
        self.cache.append([query, query_response])
        if len(self.cache) > self.max_cache_limit:
          self.cache.pop(0)

      #Check Against Verbose
        if self.verbose:
          # print("[IN EMPLOYEE CENTERED], returning: ", query_response)
          return query_response
        else:
          match = re.search(r"\*\*Provide the answer\*\*: (.*?)(?:\n|$)", query_response)
          # print("[IN EMPLOYEE CENTERED], returning: ", query_response)
          return match.group(1) if match else query_response

    #Helper functions:
    def similar(self, a, b):
        return SequenceMatcher(None, a, b).ratio()

    def rerank(self, query, chunks):
        responses = self.Cohere_client.rerank(
            query=query,
            documents=chunks,
            top_n=len(chunks) 
        )
        # print("[IN RERANK] responses are: ", responses)
        relevant_indexes = [item.index for item in responses.results]
        return [chunks[i] for i in relevant_indexes]


    # So this is what i had a theory about
    def get_relevant_docs(self,query):
        augmented_prompt = self.get_relevant_docs_chain.invoke({"question": query})
        documents = [
            "Code-of-conduct", "Compensation-Benefits-Guide", "Employee-appraisal-form",
            "Employee-Handbook", "Employee-Termination-Policy", "Health-and-Safety-Guidelines",
            "Onboarding-Manual", "Remote-Work-Policy", "Cybersecurity-for-Employees",
            "System-Access-Control-Policy", "Technology-Devices-Policy", "Expense-Report"
        ]
        words = augmented_prompt.split()
        matches = [doc for doc in documents if doc in words]
        return ", ".join(matches[:2])

    def format_docs(self, docs):
        return "\n\n".join([d.page_content for d in docs])

    def format_docs_rerank(self, docs):
      return [d.page_content for d in docs]

    def reformat_docs(self, docs):
      return "\n\n".join([d for d in docs])
