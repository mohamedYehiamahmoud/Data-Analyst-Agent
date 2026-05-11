"""
main.py — FastAPI Application Entry Point
==========================================
This is where the HTTP API lives. It:

  1. Defines routes (URL endpoints)
  2. Handles file uploads
  3. Calls the LangGraph agent
  4. Applies guardrails
  5. Manages conversation memory
  6. Returns structured responses

FastAPI automatically generates interactive docs at /docs
(Swagger UI) — open that in your browser after starting the server.

To start the server:
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file immediately
load_dotenv()

import aiofiles
import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.guardrails import sanitize_report_output, validate_csv_upload, validate_query
from src.llm_client import build_graph
from src.memory import memory_manager
from src.models import (
    AnalysisResponse,
    ColumnDescriptionResponse,
    EmailRequest,
    HealthResponse,
)
from src.email_utils import send_email_report

# ─────────────────────────────────────────────
# Logging setup
# Logs show up in your terminal when running with uvicorn.
# In production, you'd send these to a log aggregator.
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Directories
# ─────────────────────────────────────────────
UPLOAD_DIR = Path("uploads")
IMAGES_DIR = Path("images")
UPLOAD_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

APP_VERSION = "1.0.0"


# ─────────────────────────────────────────────
# Lifespan — runs at startup and shutdown
# We compile the LangGraph ONCE here so every
# request reuses the same compiled graph object.
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Building LangGraph workflow...")
    app.state.graph = build_graph()
    logger.info("✅ LangGraph workflow ready.")
    yield
    # Cleanup on shutdown (none needed for now)
    logger.info("Shutting down AutoAnalyst.")


# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="AutoAnalyst API",
    description=(
        "AI-powered CSV data analysis agent. "
        "Upload a CSV, then ask questions about your data in plain English."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

# CORS — allow all origins in development.
# In production, replace "*" with your actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated chart images as static files at /images/<filename>
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


# ─────────────────────────────────────────────
# Helper: Build column description
# ─────────────────────────────────────────────
def build_column_description(csv_path: str) -> str:
    """
    Read a CSV and return a text summary of its columns.

    Example output:
        - age: numeric (int64), range: 18 - 65
        - gender: categorical, unique values: ['M', 'F']
        - name: text, unique count: 1000

    This description is injected into LLM prompts so the model
    understands what data is available.
    """
    df = pd.read_csv(csv_path)
    lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_count = df[col].nunique()

        if dtype in ("int64", "float64"):
            col_info = f"numeric ({dtype}), range: {df[col].min()} - {df[col].max()}"
        elif unique_count < 15:
            col_info = f"categorical, unique values: {df[col].unique().tolist()}"
        else:
            col_info = f"text, unique count: {unique_count}"

        lines.append(f"- {col}: {col_info}")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Helper: Extract image filenames from report
# ─────────────────────────────────────────────
def extract_image_paths(report: str) -> list[str]:
    """
    Find all image filenames referenced in a markdown report.

    Looks for patterns like: ![title](images/abc123.png)
    Returns just the filenames: ['abc123.png']
    """
    pattern = r"!\[.*?\]\(images/([^)]+)\)"
    return re.findall(pattern, report)


# ─────────────────────────────────────────────
# Routes (API Endpoints)
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Health check endpoint.
    Returns 200 OK if the server is running.
    Useful for load balancers and monitoring tools.
    """
    return HealthResponse(status="ok", version=APP_VERSION)


@app.post("/upload", tags=["Data"])
async def upload_csv(
    file: UploadFile = File(..., description="CSV file to analyze"),
    session_id: str = Form(default_factory=lambda: str(uuid.uuid4())),
):
    """
    Upload a CSV file for analysis.

    Returns a `session_id` and `file_id` that you pass to the
    `/analyze` endpoint. Also returns a preview of the column types.

    When you upload a new CSV to an existing session, the conversation
    history is cleared (since the data context has changed).
    """
    # Validate the uploaded file
    content = await validate_csv_upload(file)

    # Save to disk with a unique name to avoid collisions
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}.csv"
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    # Build column description for preview
    column_description = build_column_description(str(file_path))

    # Clear any previous session history (new data = fresh start)
    memory_manager.clear_session(session_id)
    memory_manager.get_or_create_session(session_id, csv_filename=file.filename)

    logger.info(f"CSV uploaded: {file.filename} → {file_id}, session: {session_id}")

    return {
        "session_id": session_id,
        "file_id": file_id,
        "filename": file.filename,
        "column_preview": column_description,
        "message": "File uploaded successfully. Use file_id in /analyze requests.",
    }


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze(
    query: str = Form(..., description="Your data question in plain English"),
    file_id: str = Form(..., description="file_id returned from /upload"),
    session_id: str = Form(..., description="session_id returned from /upload"),
    max_retries: int = Form(default=5, ge=1, le=10),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Ask a question about your uploaded CSV data.

    The agent will:
    1. Check if your question is relevant to the data
    2. Rephrase it for clarity
    3. Generate Python/pandas code
    4. Security-check the code
    5. Execute it
    6. Format the results as a markdown report

    Conversation history is maintained per session_id, so follow-up
    questions can reference previous answers.
    """
    # 1. Validate query (guardrail)
    safe_query = validate_query(query)

    # 2. Verify the CSV file exists
    file_path = UPLOAD_DIR / f"{file_id}.csv"
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File '{file_id}' not found. Please upload your CSV first via /upload.",
        )

    # 3. Build column description
    try:
        column_description = build_column_description(str(file_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read CSV: {e}")

    # 4. Get conversation history for this session
    conversation_history = memory_manager.get_context(session_id)

    # 5. Build the initial LangGraph state
    initial_state = {
        "query": safe_query,
        "csv_file_path": str(file_path),
        "column_description": column_description,
        "rephrased_query": None,
        "Python_Code": None,
        "data_frame": None,
        "execution_results": None,
        "execution_error": None,
        "reports": None,
        "Python_script_check": 0,
        "max_Python_script_check": max_retries,
        "script_security_issues": None,
        "is_safe": None,
        "_terminate_workflow": False,
        "conversation_history": conversation_history,   # Memory injection
        "image_output_dir": None,                       # To be set below
    }

    # 5.5 Create session-specific image directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_image_dir_name = f"{timestamp}_{session_id}"
    session_image_dir = IMAGES_DIR / session_image_dir_name
    session_image_dir.mkdir(parents=True, exist_ok=True)
    initial_state["image_output_dir"] = session_image_dir_name

    # 6. Run the LangGraph workflow
    try:
        graph = app.state.graph
        results = graph.invoke(
            initial_state,
            config={"recursion_limit": int(os.getenv("RECURSION_LIMIT", 50))},
        )
    except Exception as e:
        logger.exception(f"Graph execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    # 7. Extract report from results
    report = results.get("reports") or results.get("execution_error") or "No output generated."
    success = "execution_error" not in results or results.get("reports") is not None

    # 8. Apply output guardrail (sanitize the report)
    logger.info(f"Report before sanitization:\n{report[:500]}...")
    report = sanitize_report_output(report)
    logger.info(f"Report after sanitization:\n{report[:500]}...")

    # 9. Save this turn to memory
    if success and report:
        memory_manager.add_turn(session_id, query=safe_query, report=report)

    # 10. Extract referenced image filenames
    images = extract_image_paths(report)

    # 11. Schedule session cleanup in the background (doesn't block response)
    background_tasks.add_task(memory_manager.cleanup_expired_sessions)

    return AnalysisResponse(
        success=success,
        report=report,
        images=images,
    )


@app.get("/describe/{file_id}", response_model=ColumnDescriptionResponse, tags=["Data"])
def describe_csv(file_id: str):
    """
    Get a description of the columns in an uploaded CSV.

    Useful for understanding what questions you can ask before
    sending an /analyze request.
    """
    file_path = UPLOAD_DIR / f"{file_id}.csv"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    df = pd.read_csv(file_path)
    col_desc = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_count = df[col].nunique()
        if dtype in ("int64", "float64"):
            col_desc[col] = f"numeric, range {df[col].min()} to {df[col].max()}"
        elif unique_count < 15:
            col_desc[col] = f"categorical: {df[col].unique().tolist()}"
        else:
            col_desc[col] = f"text, {unique_count} unique values"

    return ColumnDescriptionResponse(columns=col_desc, row_count=len(df))


@app.delete("/session/{session_id}", tags=["Memory"])
def clear_session(session_id: str):
    """
    Clear the conversation history for a session.
    """
    memory_manager.clear_session(session_id)
    return {"message": f"Session '{session_id}' cleared."}


@app.post("/email-report", tags=["Analysis"])
async def email_report(request: EmailRequest):
    """
    Send the latest report from a session to an email address.
    """
    context = memory_manager.get_context(request.session_id)
    if not context:
        raise HTTPException(status_code=404, detail="No analysis found for this session.")

    # Memory stores as "Q: ... A: ..."
    # We want the last answer (A:).
    parts = context.split("A:")
    if len(parts) < 2:
        # Fallback: maybe it's the first question and history isn't fully formatted
        # or it's a direct match.
        last_report = context.strip()
    else:
        last_report = parts[-1].strip()

    if not last_report or last_report == "[Previous conversation context]":
        raise HTTPException(status_code=404, detail="No report content found to email.")

    try:
        send_email_report(
            to_email=request.email,
            subject=f"Analysis Report: {request.session_id[:8]}",
            report_markdown=last_report
        )
        return {"message": f"Report sent successfully to {request.email}"}
    except Exception as e:
        logger.error(f"Failed to email report: {e}")
        raise HTTPException(status_code=500, detail=str(e))
