# AutoAnalyst — API Documentation

Base URL: `http://localhost:8000`
Interactive docs: `http://localhost:8000/docs`

---

## Endpoints

### `GET /health`
Health check. Returns 200 if the server is up.

**Response:**
```json
{ "status": "ok", "version": "1.0.0" }
```

---

### `POST /upload`
Upload a CSV file for analysis.

**Form fields:**
| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | Yes | CSV file (max 50 MB) |
| `session_id` | string | No | Your conversation ID. Auto-generated if omitted. |

**Response:**
```json
{
  "session_id": "my-session",
  "file_id": "abc-123-uuid",
  "filename": "sales_data.csv",
  "column_preview": "- revenue: numeric...",
  "message": "File uploaded successfully."
}
```

**Errors:**
- `400` — Not a CSV, or file is empty/corrupt
- `413` — File exceeds 50 MB limit

---

### `POST /analyze`
Ask a question about your uploaded CSV.

**Form fields:**
| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | Yes | Your question (3–1000 chars) |
| `file_id` | string | Yes | From `/upload` response |
| `session_id` | string | Yes | From `/upload` response |
| `max_retries` | int | No | Code retry limit, default 5 (1–10) |

**Response:**
```json
{
  "success": true,
  "report": "## Analysis Report\n\n...",
  "images": ["chart_uuid.png"]
}
```

**Errors:**
- `400` — Query too short/long or contains injection patterns
- `404` — file_id not found (upload first)
- `500` — Agent internal error

**Note:** Images referenced in the report are served at `/images/{filename}`.

---

### `GET /describe/{file_id}`
Preview the columns in an uploaded CSV before querying.

**Response:**
```json
{
  "columns": {
    "age": "numeric, range 18 to 65",
    "department": "categorical: ['Engineering', 'HR', 'Marketing']",
    "salary": "numeric, range 45000 to 120000"
  },
  "row_count": 1000
}
```

---

### `DELETE /session/{session_id}`
Clear conversation history for a session without deleting the uploaded file.

**Response:**
```json
{ "message": "Session 'my-session' cleared." }
```
