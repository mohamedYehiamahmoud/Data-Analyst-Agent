# AutoAnalyst — Setup & Usage Guide

AutoAnalyst is an AI-powered CSV data analysis API. You upload a CSV file, then ask questions about your data in plain English. The agent generates and executes Python/pandas code, then returns a clean markdown report.

---

## Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com) (free tier available)

---

## Quick Start

### 1. Clone the repo and enter the project

```bash
cd autoanalyst
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Open .env and paste your GROQ_API_KEY
```

### 5. Start the server

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

The API is now running at `http://localhost:8000`.
Open `http://localhost:8000/docs` in your browser for interactive Swagger docs.

---

## Usage Workflow

### Step 1 — Upload your CSV

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@your_data.csv" \
  -F "session_id=my-session"
```

Response:
```json
{
  "session_id": "my-session",
  "file_id": "abc123",
  "filename": "your_data.csv",
  "column_preview": "- age: numeric (int64), range: 18 - 65\n- salary: numeric (float64)..."
}
```

### Step 2 — Ask a question

```bash
curl -X POST http://localhost:8000/analyze \
  -F "query=What is the average salary by department?" \
  -F "file_id=abc123" \
  -F "session_id=my-session"
```

Response:
```json
{
  "success": true,
  "report": "## Analysis\n\n### Average Salary by Department\n...",
  "images": ["chart_abc123.png"]
}
```

### Step 3 — Ask follow-up questions

Reuse the same `session_id` and `file_id`. The agent remembers the last 5 Q&A turns.

```bash
curl -X POST http://localhost:8000/analyze \
  -F "query=Which department has the highest variance?" \
  -F "file_id=abc123" \
  -F "session_id=my-session"
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
autoanalyst/
├── src/
│   ├── main.py          # FastAPI app and routes
│   ├── models.py        # Pydantic schemas
│   ├── llm_client.py    # LangGraph workflow and LLM calls
│   ├── memory.py        # Conversation history
│   └── guardrails.py    # Input/output validation
├── tests/
│   ├── test_api.py
│   ├── test_guardrails.py
│   └── test_llm.py
├── docs/
├── uploads/             # Created automatically at runtime
├── images/              # Created automatically at runtime
├── .env.example
├── requirements.txt
└── pytest.ini
```
