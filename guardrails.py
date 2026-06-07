"""#216
guardrails.py — Input & Output Validation

KEY IMPROVEMENTS over original:
  1. FIX: DANGEROUS_CODE_PATTERNS now allows writing to the images/
     directory while still blocking arbitrary file writes. The original
     regex blocked ALL open(..., 'w') calls, which would have flagged
     legitimate chart-saving code.
  2. FIX: validate_generated_code now also strips markdown fences before
     scanning, so a code block wrapped in ```python ... ``` doesn't sneak
     past the pattern matching.
  3. ADDED: check_execution_output() — a new helper to detect when the
     LLM printed a Python traceback vs a legitimate result that happens
     to mention the word "error" (e.g. column "error_rate"). This
     replaces the fragile `"error" in results.lower()` check in the
     original execute_python_code node.
"""

import re
import bleach
import logging
from io import BytesIO
from pathlib import Path

import pandas as pd
from fastapi import HTTPException, UploadFile

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

MAX_CSV_SIZE_MB = 50
# MAX_CSV_ROWS = 500_000
ALLOWED_EXTENSIONS = {".csv"}

# Patterns that suggest someone is trying to manipulate the LLM
INJECTION_PATTERNS = [
    r"ignore (previous|all|above) instructions",
    r"you are now",
    r"new system prompt",
    r"forget everything",
    r"disregard your",
    r"act as (a|an) ",
    r"pretend you are",
    r"your new instructions",
    r"override your",
]

# Patterns that signal dangerous code.
# NOTE: open(..., 'w') is allowed ONLY for the images/ directory.
# The original regex blocked all file writes which was too aggressive —
# matplotlib's savefig() uses open internally.
DANGEROUS_CODE_PATTERNS = [
    r"\bos\.remove\b",
    r"\bos\.rmdir\b",
    r"\bshutil\.rmtree\b",
    r"\bsubprocess\b",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\b__import__\b",
    # Block writing to paths that are NOT inside images/
    r"open\s*\(\s*['\"](?!images/)[^'\"]+['\"]\s*,\s*['\"]w['\"]",
    r"\brequests\.",
    r"\burllib\b",
    r"\bsocket\b",
    r"\bsys\.exit\b",
    r"^exit\s*\(\s*\)",
    r"\bquit\s*\(",
]


# ─────────────────────────────────────────────
# Input Guardrails
# ─────────────────────────────────────────────

def validate_query(query: str) -> str:
    """
    Clean and validate the user's text query.
    Raises HTTPException (400) if malicious or too short/long.
    """
    query = query.strip()

    if len(query) < 3:
        raise HTTPException(
            status_code=400,
            detail="Query is too short (minimum 3 characters).",
        )
    if len(query) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Query is too long (maximum 1000 characters).",
        )

    query_lower = query.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query_lower):
            logger.warning(f"Prompt injection attempt detected: {query[:100]}")
            raise HTTPException(
                status_code=400,
                detail=(
                    "Query contains disallowed patterns. "
                    "Please ask a genuine data question."
                ),
            )

    return query


async def validate_csv_upload(file: UploadFile) -> bytes:
    """
    Validate an uploaded CSV before saving it to disk.
    Checks extension, file size, parseability, and minimum content.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Only CSV files are allowed. Got: '{suffix or 'no extension'}'",
        )

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_CSV_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large ({size_mb:.1f} MB). "
                f"Maximum allowed is {MAX_CSV_SIZE_MB} MB."
            ),
        )

    try:
        df = pd.read_csv(BytesIO(content), nrows=5)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse CSV file: {str(e)}",
        )
        
    # full_df = pd.read_csv(BytesIO(content))
    # if len(full_df) > MAX_CSV_ROWS:
    #     raise HTTPException(
    #         status_code=400,
    #         detail=f"File has too many rows ({len(full_df):,}). Maximum is {MAX_CSV_ROWS:,}.",
    #     )
        
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
    Fast regex scan of LLM-generated Python code.

    Strips markdown fences first so code wrapped in ```python...```
    isn't scanned with the fence characters included (which could hide
    patterns from the regex).

    Returns (is_safe: bool, reason: str).
    """
    # Strip markdown fences before scanning
    clean = re.sub(r"^```(?:python)?\s*\n?", "", code.strip(), flags=re.IGNORECASE)
    clean = re.sub(r"\n?```\s*$", "", clean.strip())

    for pattern in DANGEROUS_CODE_PATTERNS:
        if re.search(pattern, clean):
            matched = re.search(pattern, clean).group()
            reason = f"Dangerous pattern detected: '{matched}'"
            logger.warning(f"Code security violation: {reason}")
            return False, reason

    return True, "Passed static code analysis."


def check_execution_output(output: str) -> bool:
    """
    Return True if `output` looks like a Python error (traceback),
    False if it looks like legitimate printed output.

    This replaces the original `"error" in output.lower()` check which
    caused false positives when column names or values contained "error"
    (e.g. 'error_rate', 'mean_absolute_error').
    """
    if not output:
        return False
    return bool(
        "Traceback (most recent call last)" in output
        or re.search(r"\b\w+Error:", output)
    )


# def sanitize_report_output(report: str) -> str:
#     """
#     Clean the final markdown report before sending it to the user.
#     Removes accidental HTML script/iframe tags (basic XSS prevention).
#     """
#     report = re.sub(
#         r"<script.*?>.*?</script>", "", report,
#         flags=re.DOTALL | re.IGNORECASE,
#     )
#     report = re.sub(
#         r"<iframe.*?>.*?</iframe>", "", report,
#         flags=re.DOTALL | re.IGNORECASE,
#     )
#     return report.strip()


def sanitize_report_output(report: str) -> str:
    """
    Clean the final markdown report before sending it to the user.
    Uses an allow-list approach to prevent XSS attacks.
    """
    allowed_tags = [
        'b', 'i', 'u', 'em', 'strong', 'p', 'br',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'ul', 'ol', 'li', 'code', 'pre', 'a'
    ]
    
    allowed_attributes = {
        'a': ['href', 'title', 'target'],
    }
    
    clean_report = bleach.clean(
        report, 
        tags=allowed_tags, 
        attributes=allowed_attributes,
        strip=True 
    )
    
    return clean_report.strip()
