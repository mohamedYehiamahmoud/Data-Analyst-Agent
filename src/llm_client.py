"""#615
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
import io
import sys
import time
import logging
import uuid
from typing import Literal, Optional

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
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

from .guardrails import validate_generated_code, check_execution_output
from .models import AgentState, RelevancyGrade, SanitizingResult
from .domain_prompts import detect_domain, get_code_prompt, get_report_prompt

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# LLM Configuration
# ─────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


# ─────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────

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

REPORT_GENERATION_USER = """
Write a markdown report for the following analysis.

User Question: {query}
Analysis Output: {execution_results}

Report must include:
1. ## Summary — 2-3 sentences answering the question directly using the EXACT numbers from the output.
2. ## Key Findings — bullet points with specific numbers copied from the output above.
3. ## Charts — if any chart images were saved, reference them like: ![title](images/{image_output_dir}/filename.png)
4. ## Recommendations — 2-3 actionable insights.

IMPORTANT: Every number in the report must come from the Analysis Output above.
Do NOT invent or estimate any figures.
Format as clean markdown. Do NOT wrap in ```markdown fences.
"""

CODE_FIX_SYSTEM = """
You are a Python expert fixing pandas code.

Error type: {error_type}
Error message: {error_msg}

{extra_instructions}

Rules:
- Use only pandas, matplotlib, seaborn, uuid, os.
- Handle missing values before all operations.
- Save charts to 'images/{image_output_dir}' folder with uuid filenames, then call plt.close().
- Output ONLY the corrected Python code — no explanation, no markdown fences.
"""


# ─────────────────────────────────────────────
# Retry Helper
# ─────────────────────────────────────────────

def call_llm_with_retry(chain, inputs: dict, max_retries: int = 3, base_delay: float = 2.0):
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Rate limit hit. Retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
        except Exception:
            raise


def _get_llm(temperature: float = 0) -> ChatGroq:
    return ChatGroq(api_key=GROQ_API_KEY, temperature=temperature, model=GROQ_MODEL)


# ─────────────────────────────────────────────
# Graph Nodes
# ─────────────────────────────────────────────

def check_query_relevancy(state: AgentState) -> AgentState:
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

    next_node = "re_write_query" if result.binary_score.lower().strip() == "yes" else "query_relevancy_report"
    return {"next_node": next_node}


def query_relevancy_report(state: AgentState) -> AgentState:
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
    logger.info("NODE: generate_python_code")

    df = pd.read_csv(state["csv_file_path"])
    df_head = df.head(10).to_markdown()

    domain = detect_domain(state["column_description"])
    logger.info(f"Detected domain: {domain}")

    full_query = (
        f"{state['rephrased_query']}\n\n"
        f"Include numerical analysis AND at least one chart saved to 'images/{state['image_output_dir']}' folder.\n"
        f"Print all computed results clearly."
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(get_code_prompt(domain)),
        HumanMessagePromptTemplate.from_template("{rephrased_query}"),
    ])
    chain = prompt | _get_llm() | StrOutputParser()

    code = call_llm_with_retry(chain, {
        "df_head": df_head,
        "df_columns": state["column_description"],
        "rephrased_query": full_query,
        "image_output_dir": state["image_output_dir"],
    })
    logger.info(f"Generated code:\n{code}")
    
    
    return {
        "Python_Code": code,
        "data_frame": df,
        "domain": domain,
    }


def sanitize_python_script(state: AgentState) -> AgentState:
    logger.info("NODE: sanitize_python_script")
    code = state["Python_Code"]

    is_safe_static, reason_static = validate_generated_code(code)
    if not is_safe_static:
        logger.warning(f"Static check failed: {reason_static}")
        return {"is_safe": False, "script_security_issues": reason_static}

    ALLOWED_CONTEXT = f"""
    You are reviewing auto-generated data analysis code for a sandboxed environment.

    ALREADY PROVIDED IN SANDBOX (do not flag as undefined):
    - `df` — a pandas DataFrame, fully loaded and injected before execution
    - `pd`, `plt`, `sns`, `uuid`, `os` — all pre-imported

    EXPLICITLY ALLOWED OPERATIONS (these are required by design):
    - os.makedirs('images/...') — saving charts to the images/ folder only
    - plt.savefig('images/...') — saving chart images
    - uuid.uuid4() — generating unique filenames to avoid collisions
    - df operations without prior pd.read_csv() — df is pre-loaded

    FLAG AS UNSAFE only if the code contains:
    - os.remove(), os.rmdir(), shutil.rmtree() — file deletion
    - subprocess, eval(), exec() — system/arbitrary code execution  
    - requests, urllib, socket — network calls
    - open(..., 'w') to paths outside images/ — arbitrary file writes
    - sys.exit(), quit() — process termination
    - Any write to paths other than 'images/{state["image_output_dir"]}/'

    Review this script with the above context in mind:
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            ALLOWED_CONTEXT
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



def inject_print_statements(code: str) -> str:
    """
    Post-process LLM-generated code to guarantee all computed results are printed.

    The LLM frequently assigns results to variables without calling print():
        result = df.groupby('A')['B'].mean()   # never printed → empty output

    This function parses the AST and inserts print() after every such assignment,
    so PythonAstREPLTool captures the output and the report has real numbers.

    Also wraps any bare expression on the last line (REPL-style output).
    """
    import ast as _ast, re as _re2

    # Strip markdown fences the LLM sometimes wraps code in
    code = _re2.sub(r"^```(?:python)?\s*\n?", "", code.strip(), flags=_re2.IGNORECASE)
    code = _re2.sub(r"\n?```\s*$", "", code.strip())

    PANDAS_METHODS = {
        "mean", "sum", "count", "value_counts", "describe", "groupby",
        "corr", "agg", "aggregate", "pivot_table", "pivot", "merge",
        "sort_values", "head", "tail", "nunique", "unique", "idxmax",
        "idxmin", "max", "min", "std", "var", "median", "quantile",
        "crosstab", "cut", "qcut", "nlargest", "nsmallest", "rank",
        "resample", "rolling", "expanding", "diff", "pct_change",
    }

    def _is_pandas_call(node):
        if isinstance(node, _ast.Call):
            if isinstance(node.func, _ast.Attribute) and node.func.attr in PANDAS_METHODS:
                return True
            if isinstance(node.func, _ast.Attribute):
                return _is_pandas_call(node.func.value)
        return False

    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return code

    lines = code.splitlines()
    # Map: line number (1-based) → what to print after that line
    inserts: dict[int, str] = {}
    already_printed: set[str] = set()

    for node in tree.body:
        # Track existing print() calls
        if isinstance(node, _ast.Expr) and isinstance(node.value, _ast.Call):
            func = node.value.func
            if isinstance(func, _ast.Name) and func.id == "print":
                for arg in node.value.args:
                    if isinstance(arg, _ast.Name):
                        already_printed.add(arg.id)

        # Insert print() after assignments of pandas results
        if isinstance(node, _ast.Assign):
            for target in node.targets:
                if isinstance(target, _ast.Name):
                    vname = target.id
                    if vname not in already_printed and _is_pandas_call(node.value):
                        inserts[node.end_lineno] = vname
                        already_printed.add(vname)

    # Wrap last bare expression (if not a print/savefig/close/show)
    if tree.body:
        last = tree.body[-1]
        if isinstance(last, _ast.Expr):
            func = getattr(last.value, "func", None)
            is_print_or_plot = (
                (isinstance(func, _ast.Name) and func.id == "print")
                or (isinstance(func, _ast.Attribute) and func.attr in ("savefig", "close", "show", "tight_layout"))
            )
            if not is_print_or_plot:
                expr_src = "\n".join(lines[last.lineno - 1: last.end_lineno])
                inserts[last.end_lineno] = f"__EXPR__{expr_src}"

    if not inserts:
        return code

    result_lines = []
    for i, line in enumerate(lines, start=1):
        result_lines.append(line)
        if i in inserts:
            val = inserts[i]
            if val.startswith("__EXPR__"):
                result_lines.append(f"print({val[8:].strip()})")
            else:
                result_lines.append(f"print({val})")

    return "\n".join(result_lines)


def execute_python_code(state: AgentState) -> AgentState:
    logger.info("NODE: execute_python_code")
 
    code = state["Python_Code"]
    df = state["data_frame"]
 
    images_folder = os.path.join("images", state["image_output_dir"])
    os.makedirs(images_folder, exist_ok=True)
 
    # FIX: pass matplotlib, seaborn, uuid, and os into the sandbox
    # so the generated code can save charts correctly.
    sandbox_locals = {
        "df":   df,
        "pd":   pd,
        "plt":  plt,
        "sns":  sns,
        "uuid": uuid,
        "os":   os,
    }
 
    # Keep PythonAstREPLTool for its AST-based security checks (it parses the code
    # with ast.parse and sanitize_input before any exec happens).
    #
    # The stdout capture problem: repl._run() only wraps the LAST statement in
    # redirect_stdout — all earlier statements are exec()'d with no capture at all.
    # Fix: redirect sys.stdout to our own buffer BEFORE calling repl.run().
    #   - All-but-last statements write to sys.stdout → land in our _buf.
    #   - The last statement is wrapped by repl in its own redirect_stdout(io_buffer),
    #     and its return value comes back as the string returned by repl.run().
    # We then combine both to get the complete output.
    repl = PythonAstREPLTool(locals=sandbox_locals)
 
    code = inject_print_statements(code)
    logger.info(f"Code after print injection:\n{code}")
 
    _buf = io.StringIO()
    _old_stdout = sys.stdout
    sys.stdout = _buf
    try:
        last_line_result = repl.run(code)
    finally:
        sys.stdout = _old_stdout
 
    # _buf holds output from all-but-last statements.
    # last_line_result holds whatever the last statement printed or returned.
    mid_output = _buf.getvalue()
    last_output = last_line_result if isinstance(last_line_result, str) else ""
 
    # Avoid duplicating if repl already included mid_output in its return
    if last_output and last_output not in mid_output:
        results_text = mid_output + last_output
    else:
        results_text = mid_output or last_output
 
    logger.info(f"results:\n{results_text}")
 
    # Traceback in output → code failed, ask LLM to fix
    if check_execution_output(results_text):
        return {"execution_error": results_text, "execution_results": None}
 
    # No output at all → LLM forgot to print(), trigger retry
    if not results_text.strip():
        logger.warning("Code produced no printed output — triggering retry.")
        return {
            "execution_error": (
                "NO_OUTPUT: The code ran without errors but printed nothing. "
                "Every computed result (groupby, mean, value_counts, etc.) "
                "MUST be printed with print(). Add print() around every result variable."
            ),
            "execution_results": None,
        }
 
    logger.info(f"results_text:\n{results_text}")
    return {
        "execution_results": results_text,
        "execution_error": None,
    }
    
def re_generate_python_code(state: AgentState) -> AgentState:
    logger.info("NODE: re_generate_python_code")

    current_count = state["Python_script_check"]
    max_count = state["max_Python_script_check"]

    if current_count >= max_count:
        last_error = state.get("execution_error") or state.get("script_security_issues", "Unknown error")
        return {
            "execution_error": f"❌ Max retries ({max_count}) exceeded. Last error: {last_error}",
            "Python_script_check": current_count + 1,
            "_terminate_workflow": True,
        }

    if state.get("script_security_issues"):
        error_type = "SECURITY"
        error_msg = state["script_security_issues"]
    elif state.get("execution_error", "").startswith("NO_OUTPUT"):
        # Special case: code ran fine but produced no printed output.
        # Give a very targeted instruction instead of a generic fix prompt.
        error_type = "NO_OUTPUT"
        error_msg = state["execution_error"]
    elif state.get("execution_error"):
        error_type = "EXECUTION"
        error_msg = state["execution_error"]
    else:
        error_type = "UNKNOWN"
        error_msg = "Unknown error"

    # Targeted instructions for NO_OUTPUT: the code ran but printed nothing
    if error_type == "NO_OUTPUT":
        extra_instructions = (
            "CRITICAL: The code ran without errors but printed NOTHING.\n"
            "You MUST add print() statements around every computed result.\n"
            "Example: result = df.groupby('col')['val'].mean(); print(result)\n"
            "Every groupby/mean/sum/count/value_counts result must be printed."
        )
    else:
        extra_instructions = ""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=CODE_FIX_SYSTEM.format(
            error_type=error_type,
            error_msg=error_msg,
            image_output_dir=state["image_output_dir"],
            extra_instructions=extra_instructions,
        )),
        HumanMessage(content=f"Previous code:\n{state.get('Python_Code', '')}\n\nFix the {error_type} issue."),
    ])
    chain = prompt | _get_llm() | StrOutputParser()
    new_code = call_llm_with_retry(chain, {
        "image_output_dir": state["image_output_dir"]
    })

    return {
        "Python_Code": new_code,
        "execution_error": None,
        "script_security_issues": None,
        "is_safe": None,
        "Python_script_check": current_count + 1,
        "_terminate_workflow": False,
    }


def generate_report(state: AgentState) -> AgentState:
    logger.info("NODE: generate_report")

    df = state["data_frame"]
    df_head = df.head(10).to_markdown()

    domain = state.get("domain", "general")
    logger.info(f"Generating report with domain: {domain}")

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(get_report_prompt(domain)),
        HumanMessagePromptTemplate.from_template(REPORT_GENERATION_USER),
    ])
    chain = prompt | _get_llm() | StrOutputParser()

    report = call_llm_with_retry(chain, {
        "query": state["query"],
        "execution_results": state["execution_results"],
        "df_columns": state["column_description"],
        "df_head": df_head,
        "image_output_dir": state["image_output_dir"],
    })

    return {"reports": report}


# ─────────────────────────────────────────────
# Conditional Edge Routers
# ─────────────────────────────────────────────

def route_relevancy(state: AgentState) -> str:
    return state["next_node"]


def route_after_sanitize(state: AgentState) -> Literal["execute_python_code", "re_generate_python_code"]:
    return "execute_python_code" if state.get("is_safe") else "re_generate_python_code"


def route_after_execution(state: AgentState) -> Literal["generate_report", "re_generate_python_code", "__end__"]:
    if state.get("_terminate_workflow"):
        return END
    if state.get("execution_error"):
        return "re_generate_python_code"
    return "generate_report"


# ─────────────────────────────────────────────
# Graph Builder
# ─────────────────────────────────────────────

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("check_query_relevancy", check_query_relevancy)
    workflow.add_node("query_relevancy_report", query_relevancy_report)
    workflow.add_node("re_write_query", re_write_query)
    workflow.add_node("generate_python_code", generate_python_code)
    workflow.add_node("sanitize_python_script", sanitize_python_script)
    workflow.add_node("execute_python_code", execute_python_code)
    workflow.add_node("re_generate_python_code", re_generate_python_code)
    workflow.add_node("generate_report", generate_report)

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
