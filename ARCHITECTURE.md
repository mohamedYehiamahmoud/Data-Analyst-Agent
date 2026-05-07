# AutoAnalyst — Architecture Overview

## System Components

```
┌─────────────────────────────────────────────────────┐
│                  FastAPI HTTP Layer                   │
│  POST /upload  │  POST /analyze  │  GET /describe    │
└───────────────────────┬─────────────────────────────┘
                        │
           ┌────────────▼────────────┐
           │      Guardrails          │
           │  • Query validation      │
           │  • File type/size check  │
           │  • Injection detection   │
           └────────────┬────────────┘
                        │
           ┌────────────▼────────────┐
           │    Memory Manager        │
           │  • Session store         │
           │  • History injection     │
           │  • TTL cleanup           │
           └────────────┬────────────┘
                        │
           ┌────────────▼────────────┐
           │   LangGraph Workflow     │
           │                          │
           │  check_query_relevancy   │
           │         ↓                │
           │    re_write_query        │
           │         ↓                │
           │  generate_python_code    │
           │         ↓                │
           │  sanitize_python_script  │◄─┐
           │         ↓                │  │
           │   execute_python_code    │  │ retry
           │         ↓                │  │
           │     generate_report ─────┘  │
           │         ↓           error──►┘
           │        END                   │
           └──────────────────────────────┘
                        │
                   Groq LLM API
             (llama-3.3-70b-versatile)
```

## Data Flow for a Single Query

1. **Client** sends `POST /analyze` with `query`, `file_id`, `session_id`
2. **Guardrails** validate the query text (length, injection patterns)
3. **Memory** injects recent conversation history into the LLM context
4. **LangGraph** invokes the workflow with the initial state dict
5. Each **node** in the graph reads from state, calls the LLM, writes back partial updates
6. The **compiled graph** routes between nodes based on conditional edges
7. The **final report** is sanitized and returned to the client

## Security Layers

| Layer | What it does |
|---|---|
| Input guardrail | Regex checks for injection patterns before the LLM sees the query |
| LLM relevancy check | LLM decides if query can be answered with the CSV |
| Static code scan | Regex checks for dangerous Python patterns (os.remove, subprocess, etc.) |
| LLM code review | LLM inspects generated code for subtle security issues |
| Output sanitization | HTML script/iframe tags stripped from final report |

## Production Patterns Used

- **Retry with exponential backoff** — all LLM calls retry up to 3× on rate limit errors
- **Termination flag** — prevents infinite retry loops in the agent graph
- **Recursion limit** — LangGraph hard limit on graph traversal depth
- **Background tasks** — session cleanup runs asynchronously without blocking responses
- **Structured outputs** — Pydantic schemas enforce reliable LLM JSON responses
- **File size limits** — CSV uploads capped at 50 MB
- **Stateless graph** — the compiled LangGraph is shared across all requests safely
