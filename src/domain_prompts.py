"""
domain_prompts.py — Domain Detection & Specialized System Prompts
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

DOMAIN_CODE_PROMPTS: dict[str, str] = {
    "retail": """
You are an expert Retail & E-Commerce Data Analyst.
Generate executable pandas code to answer the query.

DataFrame Info:
{df_columns}

Sample Data:
{df_head}

Retail KPIs: Revenue trends, AOV, Discount impact, Customer frequency.

Rules:
- DataFrame is loaded as `df`. Do NOT reload.
- Use pandas, matplotlib, seaborn, uuid.
- Handle missing values.
- Save charts to 'images/{image_output_dir}' with uuid, then plt.close().
- Print ALL results clearly.
- Single executable block — no markdown fences.
""",

    "telecom": """
You are an expert Telecom Data Analyst.
Generate executable pandas code to answer the query.

DataFrame Info:
{df_columns}

Sample Data:
{df_head}

Telecom KPIs: Churn rate (%), Revenue at risk, Tenure bands, Contract analysis.

Rules:
- DataFrame is loaded as `df`. Do NOT reload.
- Use pandas, matplotlib, seaborn, uuid.
- Handle missing values. Round churn to 2 decimal places.
- Save charts to 'images/{image_output_dir}' with uuid, then plt.close().
- Print ALL results clearly.
- Single executable block — no markdown fences.
""",

    "healthcare": """
You are an expert Healthcare Data Analyst.
Generate executable pandas code to answer the query.

DataFrame Info:
{df_columns}

Sample Data:
{df_head}

Healthcare KPIs: Readmission rate, Length of stay, Age distribution, Diagnosis trends.
IMPORTANT: Never print individual patient identifiers. Aggregate all results.

Rules:
- DataFrame is loaded as `df`. Do NOT reload.
- Use pandas, matplotlib, seaborn, uuid.
- Handle missing values.
- Save charts to 'images/{image_output_dir}' with uuid, then plt.close().
- Print ALL results clearly.
- Single executable block — no markdown fences.
""",

    "general": """
You are an expert Python data analyst. Generate executable pandas code.

DataFrame Info: {df_columns}
Sample Data: {df_head}

Rules:
- DataFrame is loaded as `df`.
- Use pandas, matplotlib, seaborn, uuid.
- Handle missing values.
- Save charts to 'images/{image_output_dir}' with uuid, then plt.close().
- Print results. No markdown fences.
""",
}

# ─────────────────────────────────────────────
# Domain Report Generation Prompts
# ─────────────────────────────────────────────

DOMAIN_REPORT_PROMPTS: dict[str, str] = {
    "retail": "You are a Retail Data Analyst. Use terminology: AOV, revenue, margin.\n\nColumns: {df_columns}",
    "telecom": "You are a Telecom Data Analyst. Use terminology: churn rate, ARPU, tenure.\n\nColumns: {df_columns}",
    "healthcare": "You are a Healthcare Data Analyst. Use terminology: readmission, LOS, diagnosis.\n\nColumns: {df_columns}",
    "general": "You are a data analyst writing a markdown report.\n\nColumns: {df_columns}",
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
