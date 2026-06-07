""" #551
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
from langgraph.errors import GraphRecursionError

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
# def extract_image_paths(report: str) -> list[str]:
#     """
#     Find all image filenames referenced in a markdown report.

#     Looks for patterns like: ![title](images/abc123.png)
#     Returns just the filenames: ['abc123.png']
#     """
#     pattern = r"!\[.*?\]\(images/([^)]+)\)"
#     return re.findall(pattern, report)

def scan_session_images(session_image_dir_name: str) -> list[str]:
    """
    Return all image files actually saved in the session's image folder.
 
    Returns paths in the form  'session_dir_name/filename.png'
    so Streamlit can build the correct fetch URL.
 
    This is more reliable than parsing the LLM's markdown because:
      - The LLM sometimes forgets the session subfolder in its path.
      - The LLM sometimes adds a title attribute that breaks the regex,
        e.g.  ![chart](images/abc/uuid.png "Sales Chart")
      - The files on disk are always correct; the LLM's text is not.
    """
    
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg"}
    session_dir = Path("images") / session_image_dir_name
 
    if not session_dir.exists():
        return []
 
    return [
        f"{session_image_dir_name}/{f.name}"
        for f in sorted(session_dir.iterdir())
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ]

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
    except GraphRecursionError:
        raise HTTPException(
            status_code=500,
            detail="Agent could not generate valid code after maximum retries. Please rephrase your question."
        )        
    except Exception as e:
        logger.exception(f"Graph execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    # 7. Extract report from results
    # IMPORTANT: never expose internal error strings (NO_OUTPUT, ❌ Max retries, etc.)
    # to the user as the report. If the workflow failed, show a clean user-facing message.
    final_report = results.get("reports")
    final_error  = results.get("execution_error", "")
    terminated   = results.get("_terminate_workflow", False)

    if final_report:
        report  = final_report
        success = True
    elif terminated or not final_report:
        # Workflow exhausted retries or code never produced output.
        # Give the user an actionable message, not a raw internal error string.
        report  = (
            "## Analysis Could Not Be Completed\n\n"
            "The agent was unable to generate working code for your question after several attempts.\n\n"
            "**Suggestions:**\n"
            "- Try rephrasing your question more specifically, e.g. instead of "
            "\"What is the average value by category?\" try "
            "\"What is the average Age grouped by Gender?\"\n"
            "- Make sure your question references actual column names in the dataset.\n"
            "- Try a simpler question first to confirm the data loaded correctly."
        )
        success = False
    else:
        report  = "No output generated. Please try rephrasing your question."
        success = False

    # 8. Apply output guardrail (sanitize the report)
    logger.info(f"Report before sanitization:\n{report[:500]}...")
    report = sanitize_report_output(report)
    logger.info(f"Report after sanitization:\n{report[:500]}...")

    # 9. Save this turn to memory
    if success and report:
        memory_manager.add_turn(session_id, query=safe_query, report=report, image_dir=session_image_dir_name)

    # 10. Extract referenced image filenames
    # images = extract_image_paths(report)
    images = scan_session_images(session_image_dir_name)

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


@app.get("/email-test", tags=["Analysis"])
def email_smtp_test():
    """
    Test SMTP connectivity using the server's configured credentials.
    Call this from your browser or curl to verify email will work
    BEFORE users try to send a report.

    Returns 200 if the SMTP login succeeds, 503 with a clear reason if not.
    This never sends an actual email — it just opens and closes the connection.
    """
    import smtplib

    smtp_host     = os.getenv("SMTP_HOST")
    smtp_port     = int(os.getenv("SMTP_PORT", 587))
    smtp_user     = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    missing = [k for k, v in {
        "SMTP_HOST": smtp_host,
        "SMTP_USER": smtp_user,
        "SMTP_PASSWORD": smtp_password,
    }.items() if not v]

    if missing:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Missing SMTP environment variables: {', '.join(missing)}. "
                f"Add them to your .env file. "
                f"For Gmail, use an App Password — see: "
                f"https://myaccount.google.com/apppasswords"
            ),
        )

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_password)
        return {
            "status": "ok",
            "message": f"SMTP connection to {smtp_host}:{smtp_port} succeeded. Email is ready.",
            "sender": smtp_user,
        }
    except smtplib.SMTPAuthenticationError:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Gmail rejected the password for {smtp_user}. "
                f"Regular Gmail passwords don't work — you must use a 16-character App Password. "
                f"Generate one at: https://myaccount.google.com/apppasswords "
                f"(requires 2-Step Verification to be enabled on your account)."
            ),
        )
    except smtplib.SMTPConnectError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to {smtp_host}:{smtp_port}. Check SMTP_HOST and SMTP_PORT. Detail: {e}",
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"SMTP test failed: {str(e)}")


@app.post("/email-report", tags=["Analysis"])
async def email_report(request: EmailRequest):
    """
    Send the latest analysis report from a session to an email address.

    FIX: The original implementation parsed the context string (which is a
    truncated, formatted snippet for LLM injection) to extract the report.
    This was fragile — the context clips long reports at 1500 chars and
    reformats them. Instead we now read the last turn directly from the
    session object, which always holds the full, unmodified report.
    """
    session = memory_manager.get_session(request.session_id)
    if not session or not session.turns:
        raise HTTPException(
            status_code=404,
            detail="No analysis found for this session. Run at least one query first.",
        )

    # The last turn always holds the most recent report — full, unclipped
    last_report = session.turns[-1].report.strip()
    last_query  = session.turns[-1].query

    if not last_report:
        raise HTTPException(status_code=404, detail="Last report is empty.")

    subject = f"AutoAnalyst Report: {last_query[:60]}{'...' if len(last_query) > 60 else ''}"

    # Pre-flight: check SMTP config before attempting connection
    smtp_host = os.getenv("SMTP_HOST")
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    if not all([smtp_host, smtp_user, smtp_password]):
        raise HTTPException(
            status_code=503,
            detail=(
                "Email is not configured. "
                "Set SMTP_HOST, SMTP_USER, and SMTP_PASSWORD in your .env file to enable this feature."
            ),
        )

    try:
        send_email_report(
            to_email=request.email,
            subject=subject,
            report_markdown=last_report,
            image_dir=session.last_image_dir,   # embed charts inline in email
        )
        return {"message": f"Report sent successfully to {request.email}"}
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        err = str(e)
        logger.error(f"Failed to email report: {err}")
        # Give a specific, actionable message for the most common failure
        if "535" in err or "Username and Password not accepted" in err or "BadCredentials" in err:
            detail = (
                f"Gmail rejected the password for the configured sender account. "
                f"Regular Gmail passwords are not accepted — you must use a "
                f"16-character App Password generated at: "
                f"https://myaccount.google.com/apppasswords "
                f"(2-Step Verification must be enabled). "
                f"Set it as SMTP_PASSWORD in your .env file, then restart the server."
            )
        elif "534" in err or "2-Step" in err:
            detail = (
                "Gmail requires 2-Step Verification before App Passwords can be used. "
                "Enable it at https://myaccount.google.com/security, then generate an "
                "App Password at https://myaccount.google.com/apppasswords."
            )
        elif "Connection refused" in err or "connect" in err.lower():
            detail = f"Cannot connect to SMTP server. Check SMTP_HOST and SMTP_PORT in .env. Detail: {err}"
        else:
            detail = f"Failed to send email: {err}"
        raise HTTPException(status_code=500, detail=detail)
