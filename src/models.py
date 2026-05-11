"""
models.py — Pydantic schemas for AutoAnalyst
=============================================
This file defines ALL the data shapes used across the app:
  - API request/response bodies
  - LLM structured output schemas
  - The LangGraph agent state

Why Pydantic?
  - Automatic validation: if a field is wrong type, you get a clear error
  - Auto-generated API docs (FastAPI uses these to build /docs)
  - Structured LLM output: we tell the LLM "respond as this schema"
"""

from typing import Optional, TypedDict
from pydantic import BaseModel, Field
import pandas as pd


# ─────────────────────────────────────────────
# API Request / Response Schemas
# These are what the user sends to the API and
# what the API sends back.
# ─────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    """
    What the client sends when asking a question.
    """
    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The data analysis question to answer.",
        examples=["What is the average salary by department?"],
    )
    max_retries: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Max times to retry generating code on failure.",
    )


class EmailRequest(BaseModel):
    """
    Request body for sending a report via email.
    """
    session_id: str
    email: str = Field(..., description="Recipient email address.")


class AnalysisResponse(BaseModel):
    """
    What the API returns after finishing analysis.
    """
    success: bool = Field(description="True if analysis completed without fatal errors.")
    report: Optional[str] = Field(None, description="Markdown-formatted analysis report.")
    error: Optional[str] = Field(None, description="Error message if success=False.")
    images: list[str] = Field(
        default_factory=list,
        description="List of generated chart filenames (served from /images/).",
    )


class ColumnDescriptionResponse(BaseModel):
    """
    Returned by the /describe endpoint.
    """
    columns: dict[str, str] = Field(
        description="Column name → human-readable description."
    )
    row_count: int = Field(description="Number of rows in the dataset.")


class HealthResponse(BaseModel):
    """Simple health check response."""
    status: str
    version: str


# ─────────────────────────────────────────────
# LLM Structured Output Schemas
# ─────────────────────────────────────────────

class RelevancyGrade(BaseModel):
    """
    The LLM uses this when deciding if the user's query
    can be answered with the available columns.
    """
    binary_score: str = Field(
        description="'yes' if query is answerable with available columns, 'no' otherwise."
    )


class SanitizingResult(BaseModel):
    """
    The LLM uses this when checking if generated Python code is safe.
    """
    is_safe: bool = Field(description="True if the Python script is safe to execute.")
    reason: str = Field(description="Explanation of any safety concerns found.")


# ─────────────────────────────────────────────
# LangGraph Agent State
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    """
    The shared state that flows through the LangGraph workflow.
    """
    query: str                              # Original user question
    csv_file_path: str                      # Path to the uploaded CSV on disk
    column_description: str                 # Text description of columns (fed to LLM)
    rephrased_query: Optional[str]          # More analytical version of the query
    Python_Code: Optional[str]             # Generated pandas code
    data_frame: Optional[pd.DataFrame]     # Loaded DataFrame (in memory)
    execution_results: Optional[str]       # stdout from running the code
    execution_error: Optional[str]         # Error message if code failed
    reports: Optional[str]                 # Final markdown report
    Python_script_check: int               # How many times we've retried code
    max_Python_script_check: int           # Retry limit
    script_security_issues: Optional[str] # Security problems found in code
    next_node: Optional[str]              # Used by conditional router
    is_safe: Optional[bool]               # Result of security check
    _terminate_workflow: Optional[bool]    # Signals the graph to stop early
    conversation_history: Optional[str]   # Memory injection
    image_output_dir: Optional[str]       # Session-specific directory for images
