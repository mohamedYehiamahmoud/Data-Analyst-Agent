"""
test_api.py — API Endpoint Tests
==================================
Tests for the FastAPI HTTP layer. We use httpx's AsyncClient
to make real HTTP calls against the app without running a server.

Run with:  pytest tests/test_api.py -v
"""

import io
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock

# We import the app but mock the LLM calls so tests run without an API key
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import app


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def sample_csv_bytes():
    """A small in-memory CSV for testing."""
    data = (
        "name,age,salary,department\n"
        "Alice,30,70000,Engineering\n"
        "Bob,25,55000,Marketing\n"
        "Carol,35,90000,Engineering\n"
        "Dave,28,60000,HR\n"
        "Eve,32,85000,Engineering\n"
    )
    return data.encode("utf-8")


@pytest.fixture
def mock_graph_success():
    """Mock LangGraph to return a successful report without calling real LLM."""
    mock = MagicMock()
    mock.invoke.return_value = {
        "reports": "## Analysis\n\nAverage salary is $72,000.\n\nKey finding: Engineering pays most.",
        "execution_error": None,
    }
    return mock


@pytest.fixture
def mock_graph_failure():
    """Mock LangGraph to simulate a failed analysis."""
    mock = MagicMock()
    mock.invoke.return_value = {
        "reports": None,
        "execution_error": "Max retries exceeded.",
    }
    return mock


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────

async def get_client():
    """Return an async HTTP client wired to the FastAPI app."""
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# ─────────────────────────────────────────────
# Test: Health Check
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_check():
    """GET /health should return 200 with status=ok."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body


# ─────────────────────────────────────────────
# Test: CSV Upload
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_valid_csv(sample_csv_bytes):
    """POST /upload with a valid CSV should return 200 and a file_id."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/upload",
            files={"file": ("test.csv", io.BytesIO(sample_csv_bytes), "text/csv")},
            data={"session_id": "test-session-1"},
        )

    assert response.status_code == 200
    body = response.json()
    assert "file_id" in body
    assert "session_id" in body
    assert "column_preview" in body


@pytest.mark.asyncio
async def test_upload_wrong_extension():
    """POST /upload with a .txt file should be rejected (400)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/upload",
            files={"file": ("test.txt", io.BytesIO(b"hello world"), "text/plain")},
            data={"session_id": "test-session-2"},
        )

    assert response.status_code == 400
    assert "CSV" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_empty_csv():
    """POST /upload with an empty CSV should be rejected (400)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/upload",
            files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
            data={"session_id": "test-session-3"},
        )

    assert response.status_code == 400


# ─────────────────────────────────────────────
# Test: Describe Endpoint
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_describe_existing_file(sample_csv_bytes):
    """GET /describe/{file_id} should return column info for an uploaded file."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # First upload
        up = await client.post(
            "/upload",
            files={"file": ("test.csv", io.BytesIO(sample_csv_bytes), "text/csv")},
            data={"session_id": "test-session-desc"},
        )
        file_id = up.json()["file_id"]

        # Then describe
        response = await client.get(f"/describe/{file_id}")

    assert response.status_code == 200
    body = response.json()
    assert "columns" in body
    assert "row_count" in body
    assert body["row_count"] == 5   # our sample CSV has 5 data rows
    assert "salary" in body["columns"]


@pytest.mark.asyncio
async def test_describe_nonexistent_file():
    """GET /describe with a bad file_id should return 404."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/describe/does-not-exist")

    assert response.status_code == 404


# ─────────────────────────────────────────────
# Test: Analyze Endpoint
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analyze_success(sample_csv_bytes, mock_graph_success):
    """POST /analyze with valid inputs + mocked graph should return a report."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Upload first
        up = await client.post(
            "/upload",
            files={"file": ("test.csv", io.BytesIO(sample_csv_bytes), "text/csv")},
            data={"session_id": "test-analyze-1"},
        )
        file_id = up.json()["file_id"]

        # Inject mocked graph
        app.state.graph = mock_graph_success

        # Analyze
        response = await client.post(
            "/analyze",
            data={
                "query": "What is the average salary by department?",
                "file_id": file_id,
                "session_id": "test-analyze-1",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["report"] is not None
    assert "salary" in body["report"].lower() or "analysis" in body["report"].lower()


@pytest.mark.asyncio
async def test_analyze_missing_file():
    """POST /analyze with an invalid file_id should return 404."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/analyze",
            data={
                "query": "What is the average salary?",
                "file_id": "non-existent-id",
                "session_id": "some-session",
            },
        )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_analyze_short_query(sample_csv_bytes):
    """POST /analyze with a too-short query should be rejected (400)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        up = await client.post(
            "/upload",
            files={"file": ("test.csv", io.BytesIO(sample_csv_bytes), "text/csv")},
            data={"session_id": "test-short-q"},
        )
        file_id = up.json()["file_id"]

        response = await client.post(
            "/analyze",
            data={"query": "hi", "file_id": file_id, "session_id": "test-short-q"},
        )

    assert response.status_code == 400


# ─────────────────────────────────────────────
# Test: Session Clear
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_clear_session():
    """DELETE /session/{id} should return 200."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.delete("/session/any-session-id")

    assert response.status_code == 200
