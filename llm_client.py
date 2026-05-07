"""
llm_client.py — LLM Interaction & LangGraph Workflow
=====================================================
This is the brain of AutoAnalyst. It contains:

  1. PROMPT TEMPLATES — the text instructions we send to the LLM
  2. LLM HELPER      — retry logic, rate limit handling
  3. GRAPH NODES     — each step in the analysis pipeline
  4. GRAPH BUILDER   — assembles nodes into a LangGraph workflow

The LangGraph workflow looks like this:

  START
    │
    ▼
  check_query_relevancy ──(not relevant)──► query_relevancy_report ──► END
    │
  (relevant)
    ▼
  re_write_query
    │
    ▼
  generate_python_code
    │
    ▼
  sanitize_python_script ──(unsafe)──► re_generate_python_code
    │                                         │
  (safe)                                      │
    ▼                                         │
  execute_python_code ◄──────────────────────┘
    │
    ├──(error)──► re_generate_python_code (max retries → END)
    │
  (success)
    ▼
  generate_report ──► END
"""

import os
import time
import logging
import uuid
from typing import Literal, Optional

import pandas as pd
from groq import RateLimitError
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from guardrails import validate_generated_code
from models import AgentState, RelevancyGrade, SanitizingResult

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# LLM Configuration
# Loaded from environment variables set in .env
# ─────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


# ─────────────────────────────────────────────
# Prompt Templates
# {curly_braces} are placeholders filled in at runtime.
# ─────────────────────────────────────────────

# FIX: Much more permissive and explicit relevancy check.
# The old prompt was too strict — it said "no" for valid questions
# like "show me trends" or "summarize the data" because they didn't
# directly name a column. Now we lean toward "yes" when in doubt.
RELEVANCY_CHECK_PROMPT = """
You are a data analysis assistant. Decide if the user query can be answered
using the available DataFrame columns below.

Available Columns:
{df_columns}

User Query: {query}

Rules:
- Answer "yes" if the query is about the data, the dataset, rows, columns, values, statistics, trends, charts, summaries, or anything that could be computed from the data.
- Answer "yes" for vague questions like "tell me about the data", "show me a summary", "what are the trends".
- Answer "no" ONLY if the query is completely unrelated to data analysis (e.g. "write me a poem", "what is the capital of France").
- When in doubt, answer "yes".

Respond with ONLY "yes" or "no".
"""

REPHRASE_QUERY_PROMPT = """
Rephrase the user query into a specific, actionable data analysis instruction for pandas.

Original Query: {query}
Available Columns: {df_columns}
Recent Conversation: {history}

Rules:
- Be specific: mention column names, group-by fields, aggregation methods.
- If the query references a previous question (e.g. "and by region?"), incorporate that context.
- If the query is vague (e.g. "summarize the data"), produce a comprehensive EDA instruction.
- Output only the rephrased query — no explanation.

Rephrased Query:
"""

CODE_GENERATION_SYSTEM = """
You are an expert Python data analyst. Generate executable pandas code to answer the query.

DataFrame Info:
{df_columns}

Sample Data (first rows):
{df_head}

Rules:
- The DataFrame is already loaded as the variable `df`. Do NOT reload the CSV.
- Use ONLY pandas, matplotlib, seaborn, and uuid — no other libraries.
- Always handle missing values (dropna or fillna) before calculations.
- For any chart: use matplotlib/seaborn, save with a unique uuid filename to the 'images/' folder, then call plt.close() to free memory.
- Print ALL results so they appear in the output.
- Write as a single executable block — no functions, no classes.
- Do NOT include ```python fences or any explanation — just the code.
"""

REPORT_GENERATION_SYSTEM = """
You are an expert data analyst writing a clear markdown report from analysis results.

Dataset columns available:
{df_columns}
"""

REPORT_GENERATION_USER = """
Write a markdown report for the following analysis.

User Question: {query}
Analysis Output: {execution_results}

Report must include:
1. ## Summary — 2-3 sentences answering the question directly.
2. ## Key Findings — bullet points with specific numbers from the output.
3. ## Charts — if any chart images were saved, reference them like: ![title](images/filename.png)
4. ## Recommendations — 2-3 actionable insights.

Format as clean markdown. Do NOT wrap in ```markdown fences.
"""

CODE_FIX_SYSTEM = """
You are a Python expert fixing broken pandas code.

Error type: {error_type}
Error message: {error_msg}

Rules:
- Use only pandas, matplotlib, seaborn.
- Handle missing values before all operations.
- Save charts to 'images/' folder with uuid filenames.
- Output ONLY the corrected Python code — no explanation, no markdown fences.
"""


# ─────────────────────────────────────────────
# Retry Helper
#
# LLM APIs sometimes return "rate limit" errors when overloaded.
# We wait and retry instead of crashing. Each wait is 2x longer
# than the previous one (exponential backoff).
# ─────────────────────────────────────────────

def call_llm_with_retry(chain, inputs: dict, max_retries: int = 3, base_delay: float = 2.0):
    """
    Call a LangChain chain, retrying up to max_retries times on rate limit errors.

    Example timing with base_delay=2:
      Attempt 1 fails → wait 2s
      Attempt 2 fails → wait 4s
      Attempt 3 fails → raise error
    """
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except RateLimitError:
            if attempt == max_retries - 1:
                raise  # No retries left — give up
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Rate limit hit. Retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
        except Exception:
            raise  # Any other error fails immediately — no point retrying


def _get_llm(temperature: float = 0) -> ChatGroq:
    """Create a Groq LLM instance. temperature=0 gives consistent, deterministic output."""
    return ChatGroq(api_key=GROQ_API_KEY, temperature=temperature, model=GROQ_MODEL)


# ─────────────────────────────────────────────
# Graph Nodes
# Each function = one step in the workflow.
# They receive the full state dict and return ONLY the keys they changed.
# ─────────────────────────────────────────────

def check_query_relevancy(state: AgentState) -> AgentState:
    """
    NODE 1: Is the user's question related to the CSV data?

    Uses structured output (RelevancyGrade) so we get a clean yes/no
    instead of having to parse free-form LLM text.
    """
    logger.info("NODE: check_query_relevancy")

    prompt = PromptTemplate(
        template=RELEVANCY_CHECK_PROMPT,
        input_variables=["df_columns", "query"],
    )
    llm = _get_llm().with_structured_output(RelevancyGrade)
    chain = prompt | llm

    result = call_llm_with_retry(chain, {
        "df_columns": state["column_description"],
        "query": state["query"],
    })

    logger.info(f"Relevancy result: {result.binary_score!r} for query: {state['query']!r}")

    # Route to the right next node
    next_node = "re_write_query" if result.binary_score.lower().strip() == "yes" else "query_relevancy_report"
    return {"next_node": next_node}


def query_relevancy_report(state: AgentState) -> AgentState:
    """
    NODE 2a (branch): Return a polite message when the query is off-topic.
    """
    logger.info("NODE: query_relevancy_report")
    return {
        "reports": (
            f"## Query Not Relevant to the Data\n\n"
            f"Your question **'{state['query']}'** doesn't appear to be answerable "
            f"with the available columns:\n\n"
            f"{state['column_description']}\n\n"
            f"**Try asking something like:**\n"
            f"- What is the average value per category?\n"
            f"- Show me the top 10 rows by value.\n"
            f"- Are there any missing values?\n"
            f"- What are the trends over time?"
        )
    }


def re_write_query(state: AgentState) -> AgentState:
    """
    NODE 2b: Rephrase the user's vague question into a specific analysis instruction.

    Example:
      User:  "show me sales"
      LLM:   "Calculate total and average sales by product category,
               sort descending, show top 10."

    This makes code generation much more reliable.
    """
    logger.info("NODE: re_write_query")

    prompt = PromptTemplate(
        template=REPHRASE_QUERY_PROMPT,
        input_variables=["query", "df_columns", "history"],
    )
    chain = prompt | _get_llm() | StrOutputParser()

    rephrased = call_llm_with_retry(chain, {
        "query": state["query"],
        "df_columns": state["column_description"],
        "history": state.get("conversation_history", "No previous conversation."),
    })

    logger.info(f"Rephrased query: {rephrased}")
    return {"rephrased_query": rephrased}


def generate_python_code(state: AgentState) -> AgentState:
    """
    NODE 3: Generate Python/pandas code to answer the rephrased query.

    The LLM receives column descriptions + sample rows so it knows
    the exact column names and data types to use in the code.
    """
    logger.info("NODE: generate_python_code")

    # Load the CSV (needed for the data sample in the prompt)
    df = pd.read_csv(state["csv_file_path"])

    # FIX: Use df.head() instead of df.sample() so the LLM always sees
    # a predictable slice of data (sample() can cause confusion with indexes)
    df_head = df.head(10).to_markdown()

    # Combine the rephrased query with code-writing instructions
    full_query = (
        f"{state['rephrased_query']}\n\n"
        f"Include numerical analysis AND at least one chart saved to 'images/' folder.\n"
        f"Print all computed results clearly."
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(CODE_GENERATION_SYSTEM),
        HumanMessagePromptTemplate.from_template("{rephrased_query}"),
    ])
    chain = prompt | _get_llm() | StrOutputParser()

    code = call_llm_with_retry(chain, {
        "df_head": df_head,
        "df_columns": state["column_description"],
        "rephrased_query": full_query,
    })

    return {
        "Python_Code": code,
        "data_frame": df,  # Store df in state so later nodes don't re-read the CSV
    }


def sanitize_python_script(state: AgentState) -> AgentState:
    """
    NODE 4: Two-layer security check on the generated code.

    Layer 1 — Fast regex scan (guardrails.py): catches obvious dangers
              like os.remove(), subprocess, eval() in microseconds.

    Layer 2 — LLM security review: catches subtler issues like
              disguised dangerous operations or infinite loops.

    Both must pass before the code is executed.
    """
    logger.info("NODE: sanitize_python_script")
    code = state["Python_Code"]

    # Layer 1: fast static check (no LLM call needed)
    is_safe_static, reason_static = validate_generated_code(code)
    if not is_safe_static:
        logger.warning(f"Static check failed: {reason_static}")
        return {"is_safe": False, "script_security_issues": reason_static}

    # Layer 2: LLM security review
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a Python security expert. "
            "Check if this script is safe: no file deletion, system calls, "
            "network requests, infinite loops, or any destructive operations."
        )),
        HumanMessage(content=f"Python script to review:\n{code}"),
    ])
    llm = _get_llm().with_structured_output(
        schema=SanitizingResult,
        method="function_calling",
        include_raw=False,
    )
    sanitize_chain = prompt | llm
    response = call_llm_with_retry(sanitize_chain, {"input": ""})

    if response.is_safe:
        return {"is_safe": True, "script_security_issues": None}
    else:
        logger.warning(f"LLM security check failed: {response.reason}")
        return {"is_safe": False, "script_security_issues": response.reason}


def execute_python_code(state: AgentState) -> AgentState:
    """
    NODE 5: Run the sanitized code in a restricted sandbox.

    PythonAstREPLTool runs the code with ONLY the `df` variable available.
    Any print() output becomes the execution_results that feeds the report.
    """
    logger.info("NODE: execute_python_code")

    code = state["Python_Code"]
    df = state["data_frame"]

    # The sandbox only has access to df and pd — nothing else
    repl = PythonAstREPLTool(locals={"df": df, "pd": pd})

    # Make sure charts have somewhere to be saved
    os.makedirs("images", exist_ok=True)

    try:
        results = repl.run(code)

        # If the output contains "error", treat it as a runtime failure
        if results and "error" in results.lower():
            return {"execution_error": results, "execution_results": None}

        return {
            "execution_results": results or "Code ran successfully (no printed output).",
            "execution_error": None,
        }
    except Exception as e:
        logger.error(f"Code execution error: {e}")
        return {"execution_error": str(e), "execution_results": None}


def re_generate_python_code(state: AgentState) -> AgentState:
    """
    NODE 6: Ask the LLM to fix the broken code.

    Handles both security failures and execution errors.
    Tracks retry count and stops when max retries are hit.
    """
    logger.info("NODE: re_generate_python_code")

    current_count = state["Python_script_check"]
    max_count = state["max_Python_script_check"]

    # If we've hit the retry limit, stop and report the last error
    if current_count >= max_count:
        last_error = state.get("execution_error") or state.get("script_security_issues", "Unknown error")
        return {
            "execution_error": f"❌ Max retries ({max_count}) exceeded. Last error: {last_error}",
            "Python_script_check": current_count + 1,
            "_terminate_workflow": True,
        }

    # Figure out what kind of error to tell the LLM about
    if state.get("script_security_issues"):
        error_type = "SECURITY"
        error_msg = state["script_security_issues"]
    elif state.get("execution_error"):
        error_type = "EXECUTION"
        error_msg = state["execution_error"]
    else:
        error_type = "UNKNOWN"
        error_msg = "Unknown error"

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=CODE_FIX_SYSTEM.format(error_type=error_type, error_msg=error_msg)),
        HumanMessage(content=f"Previous code:\n{state.get('Python_Code', '')}\n\nFix the {error_type} issue."),
    ])
    chain = prompt | _get_llm() | StrOutputParser()
    new_code = call_llm_with_retry(chain, {})

    return {
        "Python_Code": new_code,
        "execution_error": None,
        "script_security_issues": None,
        "is_safe": None,
        "Python_script_check": current_count + 1,
        "_terminate_workflow": False,
    }


def generate_report(state: AgentState) -> AgentState:
    """
    NODE 7: Turn raw execution output into a clean markdown report.

    The LLM formats the printed results into an executive summary,
    key findings, chart references, and recommendations.
    """
    logger.info("NODE: generate_report")

    df = state["data_frame"]
    # FIX: Use df.head() for consistency (same as code generation node)
    df_head = df.head(10).to_markdown()

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(REPORT_GENERATION_SYSTEM),
        HumanMessagePromptTemplate.from_template(REPORT_GENERATION_USER),
    ])
    chain = prompt | _get_llm() | StrOutputParser()

    report = call_llm_with_retry(chain, {
        "query": state["query"],
        "execution_results": state["execution_results"],
        "df_columns": state["column_description"],
        "df_head": df_head,
    })

    return {"reports": report}


# ─────────────────────────────────────────────
# Conditional Edge Routers
# These look at the state and return the name of the next node.
# ─────────────────────────────────────────────

def route_relevancy(state: AgentState) -> str:
    """Go to re_write_query if relevant, otherwise show a refusal message."""
    return state["next_node"]


def route_after_sanitize(state: AgentState) -> Literal["execute_python_code", "re_generate_python_code"]:
    """Run code if safe; regenerate if security check failed."""
    return "execute_python_code" if state.get("is_safe") else "re_generate_python_code"


def route_after_execution(state: AgentState) -> Literal["generate_report", "re_generate_python_code", "__end__"]:
    """Generate report if execution succeeded; retry or stop if it failed."""
    if state.get("_terminate_workflow"):
        return END
    if state.get("execution_error"):
        return "re_generate_python_code"
    return "generate_report"


# ─────────────────────────────────────────────
# Graph Builder
# Wires all nodes into the workflow and compiles it.
# Call this ONCE at startup — the compiled graph is reusable and thread-safe.
# ─────────────────────────────────────────────

def build_graph():
    """
    Assemble and compile the LangGraph workflow.

    Usage:
        graph = build_graph()
        result = graph.invoke(initial_state, config={"recursion_limit": 50})
    """
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("check_query_relevancy", check_query_relevancy)
    workflow.add_node("query_relevancy_report", query_relevancy_report)
    workflow.add_node("re_write_query", re_write_query)
    workflow.add_node("generate_python_code", generate_python_code)
    workflow.add_node("sanitize_python_script", sanitize_python_script)
    workflow.add_node("execute_python_code", execute_python_code)
    workflow.add_node("re_generate_python_code", re_generate_python_code)
    workflow.add_node("generate_report", generate_report)

    # Wire edges
    workflow.add_edge(START, "check_query_relevancy")
    workflow.add_conditional_edges("check_query_relevancy", route_relevancy)
    workflow.add_edge("query_relevancy_report", END)
    workflow.add_edge("re_write_query", "generate_python_code")
    workflow.add_edge("generate_python_code", "sanitize_python_script")
    workflow.add_conditional_edges("sanitize_python_script", route_after_sanitize)
    workflow.add_conditional_edges("execute_python_code", route_after_execution)
    workflow.add_edge("re_generate_python_code", "sanitize_python_script")
    workflow.add_edge("generate_report", END)

    return workflow.compile()
