import re
from uuid import uuid4
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import TokenTextSplitter

rag_prompt_template = """Answer the question based on the given context and question.

Context: {context} Question: {question} Answer:"""

rag_prompt = PromptTemplate(
    template=rag_prompt_template,
    input_variables=["context", "question"],
)

class DocumentLoader:
    def __init__(self, chunk_size=256, chunk_overlap=0.50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _load_and_split(self, docs_to_load):
        combined_text = ""
        for doc in docs_to_load:
            loader = PyMuPDFLoader(doc)
            documents = loader.load()
            for page in documents:
                text = page.page_content
                if "contents" in text.lower():
                    continue
                text = re.sub(r"\bPage\s+\d+\b", "", text, flags=re.IGNORECASE)
                text = re.sub(r"\n", "", text).strip()
                text = re.sub(r"[^\w\s.,?!:;\'\"()&-]", "", text)
                combined_text += text + " "
        combined_text = combined_text.strip()

        text_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=int(self.chunk_size * self.chunk_overlap),
        )
        texts = text_splitter.split_text(combined_text)
        return text_splitter.create_documents(texts)


class VectorStore:
    def __init__(self, pinecone_client, index_name, embedding_model):
        self.pc = pinecone_client
        self.index_name = index_name
        self.embeddings = embedding_model

        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        self.index = self.pc.Index(self.index_name)
        self.vector_store = PineconeVectorStore(
            index=self.index, embedding=self.embeddings
        )

    def _build(self, documents):
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vector_store.add_documents(documents=documents, ids=uuids)

    def _as_retriever(
        self, search_type="similarity_score_threshold", search_kwargs=None
    ):
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs or {"k": 3, "score_threshold": 0.5},
        )


class CustomerRagAgent:
    def __init__(self, document_loader, vector_store, llm):
        self.document_loader = document_loader
        self.vector_store = vector_store
        self.retriever = self.vector_store._as_retriever()
        self.llm = llm
        self.rag_chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | rag_prompt
            | self.llm
        )

    def _format_docs(self, docs):
        return "\n\n".join([d.page_content for d in docs])

    def load_and_add_documents(self, docs_to_load):
        documents = self.document_loader._load_and_split(docs_to_load)
        self.vector_store._build(documents)

    def generate(self, query):
        query_response = self.rag_chain.invoke(query)
        return query_response
