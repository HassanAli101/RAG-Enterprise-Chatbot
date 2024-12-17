from pydantic import BaseModel, Field

class ToSqlAgent(BaseModel):
    """A tool to route the current dialog to the SQL agent."""
    query: str = Field(description="The user question to be passed to the SQL agent.")
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the status of order number 3?",
            },
        }

class ToRagAgent(BaseModel):
    """A tool to route the current dialog to the RAG agent."""
    query: str = Field(description="The query to be passed to the RAG agent for document-based information retrieval.")
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the refund policy for Uber Eats?",
            },
        }

class ToRepresentative(BaseModel):
    """A tool to route the current dialog to the customer service representative for final response to the user."""
