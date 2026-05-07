"""
guardrails.py — Input & Output Validation
==========================================
Guardrails are safety checks that run BEFORE and AFTER
the LLM to catch bad inputs or dangerous outputs early.

Think of them as airport security:
  - Input guardrails = check what goes IN to the LLM
  - Output guardrails = check what comes OUT of the LLM

Why guardrails?
  - Prevent prompt injection (users trying to hijack the LLM)
  - Block clearly off-topic or abusive queries
  - Validate uploaded files before processing
  - Catch dangerous patterns in generated code
"""

import re
import os
import logging
from pathlib import Path

import pandas as pd
from fastapi import HTTPException, UploadFile

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

MAX_CSV_SIZE_MB = 50
MAX_CSV_ROWS = 500_000
ALLOWED_EXTENSIONS = {".csv"}

# Patterns that suggest someone is trying to manipulate the LLM
INJECTION_PATTERNS = [
    r"ignore (previous|all|above) instructions",
    r"you are now",
    r"new system prompt",
    r"forget everything",
    r"disregard your",
    r"act as (a|an) ",
]

# Patterns that signal dangerous code — things generated code should never do
DANGEROUS_CODE_PATTERNS = [
    r"\bos\.remove\b",
    r"\bos\.rmdir\b",
    r"\bshutil\.rmtree\b",
    r"\bsubprocess\b",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\b__import__\b",
    r"\bopen\s*\(.*['\"]w['\"]",   # writing to arbitrary files
    r"\brequests\.",               # network calls
    r"\burllib\b",
    r"\bsocket\b",
]


# ─────────────────────────────────────────────
# Input Guardrails
# ─────────────────────────────────────────────

def validate_query(query: str) -> str:
    """
    Clean and validate the user's text query.

    Raises HTTPException (400) if the query looks malicious or too short.
    Returns the stripped query if it passes.

    This runs BEFORE the query reaches the LLM.
    """
    # Strip whitespace
    query = query.strip()

    if len(query) < 3:
        raise HTTPException(status_code=400, detail="Query is too short (min 3 characters).")

    if len(query) > 1000:
        raise HTTPException(status_code=400, detail="Query is too long (max 1000 characters).")

    # Check for prompt injection attempts
    query_lower = query.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query_lower):
            logger.warning(f"Prompt injection attempt detected: {query[:100]}")
            raise HTTPException(
                status_code=400,
                detail="Query contains disallowed patterns. Please ask a genuine data question.",
            )

    return query


async def validate_csv_upload(file: UploadFile) -> bytes:
    """
    Validate an uploaded CSV file before saving it to disk.

    Checks:
      1. File extension must be .csv
      2. File size must be under MAX_CSV_SIZE_MB
      3. Content must be parseable as a CSV
      4. Must have at least 1 column and 1 row

    Returns the raw file bytes if valid.
    Raises HTTPException on any failure.
    """
    # 1. Extension check
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Only CSV files are allowed. Got: '{suffix or 'no extension'}'",
        )

    # 2. Read bytes and check size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_CSV_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum is {MAX_CSV_SIZE_MB} MB.",
        )

    # 3. Parse CSV to check validity
    try:
        import io
        df = pd.read_csv(io.BytesIO(content), nrows=5)  # Only read 5 rows to validate
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse CSV file: {str(e)}",
        )

    # 4. Check it has usable data
    if df.empty or len(df.columns) == 0:
        raise HTTPException(
            status_code=400,
            detail="CSV file is empty or has no columns.",
        )

    return content


# ─────────────────────────────────────────────
# Output Guardrails
# ─────────────────────────────────────────────

def validate_generated_code(code: str) -> tuple[bool, str]:
    """
    Quick rule-based scan of LLM-generated Python code.

    This is a FAST first pass using regex — it runs in microseconds.
    The LLM-based security check (in llm_client.py) runs after this
    as a second, deeper layer.

    Returns:
        (is_safe: bool, reason: str)
    """
    for pattern in DANGEROUS_CODE_PATTERNS:
        if re.search(pattern, code):
            matched = re.search(pattern, code).group()
            reason = f"Dangerous pattern detected: '{matched}'"
            logger.warning(f"Code security violation: {reason}")
            return False, reason

    return True, "Passed static code analysis."


def sanitize_report_output(report: str) -> str:
    """
    Clean the final markdown report before sending it to the user.

    Removes any accidental HTML script tags or other potentially
    dangerous content that the LLM might hallucinate.
    """
    # Remove <script> tags (basic XSS prevention)
    report = re.sub(r"<script.*?>.*?</script>", "", report, flags=re.DOTALL | re.IGNORECASE)
    # Remove <iframe> tags
    report = re.sub(r"<iframe.*?>.*?</iframe>", "", report, flags=re.DOTALL | re.IGNORECASE)
    return report.strip()
