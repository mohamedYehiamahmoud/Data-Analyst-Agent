"""#276
domain_prompts.py — Domain Detection & Specialized System Prompts

KEY IMPROVEMENT over original:
  The {execution_results} placeholder in code prompts was included in the
  system prompt template but was always passed as an empty string on first
  generation (because we haven't run the code yet). This confused the LLM.

  Fix: The placeholder is now removed from code-generation prompts. It was
  only meaningful at report time, and the report prompt (in llm_client.py)
  already handles it correctly. The code prompt's job is to GENERATE code
  that PRODUCES results — not to consume results.
"""

from typing import Literal

DomainType = Literal["retail", "telecom", "healthcare", "general"]

# ─────────────────────────────────────────────
# Domain Signal Keywords
# ─────────────────────────────────────────────

DOMAIN_SIGNALS: dict[str, list[str]] = {
    "retail": [
        "product", "category", "sales", "revenue", "order", "customer",
        "quantity", "price", "discount", "profit", "store", "sku",
        "invoice", "purchase", "item", "basket", "transaction",
    ],
    "telecom": [
        "churn", "contract", "tenure", "monthly_charges", "total_charges",
        "internet_service", "phone_service", "streaming", "tech_support",
        "payment_method", "paperless", "senior_citizen", "partner",
        "dependents", "multiple_lines", "online_security",
    ],
    "healthcare": [
        "patient", "diagnosis", "icd", "procedure", "admission",
        "discharge", "bmi", "blood_pressure", "cholesterol", "glucose",
        "outcome", "readmission", "length_of_stay", "insurance",
        "provider", "medication",
    ],
}

# ─────────────────────────────────────────────
# Domain Code Generation Prompts
# ─────────────────────────────────────────────
# NOTE: {execution_results} has been REMOVED from these prompts.
# At code-generation time there are no results yet — injecting an empty
# placeholder confused the model. Results are used only in the report step.
# ─────────────────────────────────────────────

DOMAIN_CODE_PROMPTS: dict[str, str] = {
    "retail": """
You are an expert Retail & E-Commerce Data Analyst.
Generate executable pandas code to answer the query.

CRITICAL RULE — READ FIRST:
The DataFrame is ALREADY loaded in memory as the variable `df`.
DO NOT use pd.read_csv(). DO NOT use open(). DO NOT reference any filename.
Start your code directly with `df` operations.

WRONG (never do this):
  df = pd.read_csv('data.csv')
  df = pd.DataFrame(data)

CORRECT (always do this):
  result = df.groupby(...)
  print(result)

DataFrame column info:
{df_columns}

Sample rows (first 10):
{df_head}

Retail KPIs to consider: Revenue trends, AOV, Discount impact, Customer frequency.

Rules:
- Use pandas, matplotlib, seaborn, uuid, os.
- Handle missing values (dropna / fillna) before all operations.
- Save charts to 'images/{image_output_dir}/' with uuid filenames, then call plt.close().
- PRINT ALL computed results clearly — they will be quoted in a report.
- Single executable block — no markdown fences, no explanation text.
""",

    "telecom": """
You are an expert Telecom Data Analyst.
Generate executable pandas code to answer the query.

CRITICAL RULE — READ FIRST:
The DataFrame is ALREADY loaded in memory as the variable `df`.
DO NOT use pd.read_csv(). DO NOT use open(). DO NOT reference any filename.
Start your code directly with `df` operations.

WRONG (never do this):
  df = pd.read_csv('data.csv')
  df = pd.DataFrame(data)

CORRECT (always do this):
  result = df.groupby(...)
  print(result)

DataFrame column info:
{df_columns}

Sample rows (first 10):
{df_head}

Telecom KPIs to consider: Churn rate (%), Revenue at risk, Tenure bands, Contract analysis.

Rules:
- Use pandas, matplotlib, seaborn, uuid, os.
- Handle missing values. Round churn percentages to 2 decimal places.
- Save charts to 'images/{image_output_dir}/' with uuid filenames, then call plt.close().
- PRINT ALL computed results clearly — they will be quoted in a report.
- Single executable block — no markdown fences, no explanation text.
""",

    "healthcare": """
You are an expert Healthcare Data Analyst.
Generate executable pandas code to answer the query.

CRITICAL RULE — READ FIRST:
The DataFrame is ALREADY loaded in memory as the variable `df`.
DO NOT use pd.read_csv(). DO NOT use open(). DO NOT reference any filename.
Start your code directly with `df` operations.

WRONG (never do this):
  df = pd.read_csv('data.csv')
  df = pd.DataFrame(data)

CORRECT (always do this):
  result = df.groupby(...)
  print(result)

DataFrame column info:
{df_columns}

Sample rows (first 10):
{df_head}

Healthcare KPIs to consider: Readmission rate, Length of stay, Age distribution, Diagnosis trends.

IMPORTANT PRIVACY RULE: Never print individual patient identifiers.
Always aggregate results (groupby, mean, count, etc.).

Rules:
- Use pandas, matplotlib, seaborn, uuid, os.
- Handle missing values.
- Save charts to 'images/{image_output_dir}/' with uuid filenames, then call plt.close().
- PRINT ALL computed results clearly — they will be quoted in a report.
- Single executable block — no markdown fences, no explanation text.
""",

    "general": """
You are an expert Python data analyst.
Generate executable pandas code to answer the query.

CRITICAL RULE — READ FIRST:
The DataFrame is ALREADY loaded in memory as the variable `df`.
DO NOT use pd.read_csv(). DO NOT use open(). DO NOT reference any filename.
Start your code directly with `df` operations.

WRONG (never do this):
  df = pd.read_csv('data.csv')
  df = pd.DataFrame(data)

CORRECT (always do this):
  result = df.groupby(...)
  print(result)

DataFrame column info:
{df_columns}

Sample rows (first 10):
{df_head}

Rules:
- Use pandas, matplotlib, seaborn, uuid, os.
- Handle missing values.
- Save charts to 'images/{image_output_dir}/' with uuid filenames, then call plt.close().
- PRINT ALL computed results clearly — they will be quoted in a report.
- Single executable block — no markdown fences, no explanation text.
""",
}

# ─────────────────────────────────────────────
# Domain Report Generation Prompts
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# ANTI-HALLUCINATION REPORT SYSTEM PROMPTS
#
# KEY RULE enforced here:
#   The {df_columns} block describes column *metadata* (dtypes, ranges, unique
#   counts). It is provided so the analyst understands the schema — NOT as a
#   source of result figures. Any number that appears only in {df_columns} and
#   NOT in the AUTHORITATIVE DATA block must NOT be reported as an analysis
#   result.  The previous version of these prompts didn't make this distinction,
#   so the LLM reported metadata stats (e.g. "unique count: 92") as if they
#   were computed analysis results.
# ─────────────────────────────────────────────────────────────────────────────

_SHARED_ANTI_HALLUCINATION_RULES = """
CRITICAL ANTI-HALLUCINATION RULES:
1. The AUTHORITATIVE DATA block in the user message is the ONLY valid source
   of numbers for your report. Every figure you write must appear there verbatim.
2. The "Columns" section below describes dataset schema (dtypes, value ranges,
   unique counts). It is for context ONLY — do NOT quote those metadata numbers
   as analysis results.
3. If the AUTHORITATIVE DATA block does not contain a figure needed to answer
   the question, write "not available in this analysis" — never invent or
   approximate a value.
4. Do NOT interpret column metadata (e.g. "unique count: 92", "range: 18-24")
   as computed analysis results. Those are schema descriptions, not findings.

Dataset schema (for context only — do NOT report these as results):
{df_columns}
"""

DOMAIN_REPORT_PROMPTS: dict[str, str] = {
    "retail": (
        "You are a Retail Data Analyst writing a concise business report. "
        "Use domain terminology: AOV, revenue, margin, basket size, conversion rate. "
        + _SHARED_ANTI_HALLUCINATION_RULES
    ),
    "telecom": (
        "You are a Telecom Data Analyst writing a concise business report. "
        "Use domain terminology: churn rate, ARPU, tenure, MRR, contract type. "
        + _SHARED_ANTI_HALLUCINATION_RULES
    ),
    "healthcare": (
        "You are a Healthcare Data Analyst writing a concise clinical report. "
        "Use domain terminology: readmission rate, LOS, diagnosis group, comorbidity. "
        "Never identify individual patients — always aggregate. "
        + _SHARED_ANTI_HALLUCINATION_RULES
    ),
    "general": (
        "You are a data analyst writing a clear, accurate markdown report. "
        + _SHARED_ANTI_HALLUCINATION_RULES
    ),
}

# ─────────────────────────────────────────────
# Core Scoring Logic
# ─────────────────────────────────────────────

def _score_text(text: str) -> DomainType:
    scores: dict[str, int] = {"retail": 0, "telecom": 0, "healthcare": 0}
    for domain, signals in DOMAIN_SIGNALS.items():
        for signal in signals:
            if signal in text:
                scores[domain] += 1
    best_domain = max(scores, key=lambda d: scores[d])
    return best_domain if scores[best_domain] >= 2 else "general"


# ─────────────────────────────────────────────
# Domain Detection Functions
# ─────────────────────────────────────────────

def detect_domain_from_columns(column_headers: list[str]) -> DomainType:
    col_text = " ".join(col.lower().replace(" ", "_") for col in column_headers)
    return _score_text(col_text)


def detect_domain(column_description: str) -> DomainType:
    return _score_text(column_description.lower())


def get_code_prompt(domain: DomainType) -> str:
    return DOMAIN_CODE_PROMPTS.get(domain, DOMAIN_CODE_PROMPTS["general"])


def get_report_prompt(domain: DomainType) -> str:
    return DOMAIN_REPORT_PROMPTS.get(domain, DOMAIN_REPORT_PROMPTS["general"])
