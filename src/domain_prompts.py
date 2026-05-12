"""
domain_prompts.py — Domain Detection & Specialized System Prompts
==================================================================
Detects the data domain from column names and returns a domain-specific
system prompt for code generation and report writing.

Supported domains: retail, telecom, healthcare, general (fallback)
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

Retail KPIs to consider when relevant:
- Revenue / Sales trends by time period or category
- Top 10 products by revenue and profit
- Average Order Value (AOV) = total revenue / number of orders
- Discount impact on profit margins
- Customer purchase frequency

Rules:
- The DataFrame is already loaded as `df`. Do NOT reload the CSV.
- Use ONLY pandas, matplotlib, seaborn, and uuid.
- Handle missing values (dropna or fillna) before calculations.
- For any chart: save with a unique uuid filename to 'images/{image_output_dir}', then call plt.close().
- Print ALL results clearly with labels.
- Write as a single executable block — no functions, no classes.
- Do NOT include ```python fences — just the code.
""",

    "telecom": """
You are an expert Telecom Data Analyst specializing in churn analysis and customer lifetime value.
Generate executable pandas code to answer the query.

DataFrame Info:
{df_columns}

Sample Data:
{df_head}

Telecom KPIs to consider when relevant:
- Churn rate = (churned customers / total customers) x 100  — always express as %
- Revenue at risk = sum of monthly_charges of churned customers
- Average tenure of churned vs retained customers
- Churn rate by contract type (Month-to-month vs annual)
- Churn rate by internet service type
- Customer segmentation by tenure bands (0-12, 12-24, 24+ months)

Common column meanings:
- churn / churned: whether customer left (Yes/No or 1/0)
- tenure: months with the company
- monthly_charges: monthly bill
- contract: contract type

Rules:
- The DataFrame is already loaded as `df`. Do NOT reload the CSV.
- Use ONLY pandas, matplotlib, seaborn, and uuid.
- Handle missing values (dropna or fillna) before calculations.
- For churn rate: always round to 2 decimal places.
- For any chart: save with a unique uuid filename to 'images/{image_output_dir}', then call plt.close().
- Print ALL results clearly with labels.
- Write as a single executable block — no functions, no classes.
- Do NOT include ```python fences — just the code.
""",

    "healthcare": """
You are an expert Healthcare Data Analyst.
Generate executable pandas code to answer the query.

DataFrame Info:
{df_columns}

Sample Data:
{df_head}

Healthcare KPIs to consider when relevant:
- Readmission rate = (readmitted patients / total patients) x 100
- Average length of stay overall and by diagnosis
- Age distribution using bins: 0-18, 18-40, 40-60, 60+
- Top 10 most common diagnoses
- Outcome distribution by age group or gender

IMPORTANT: Never print individual patient identifiers in output. Aggregate all results.

Rules:
- The DataFrame is already loaded as `df`. Do NOT reload the CSV.
- Use ONLY pandas, matplotlib, seaborn, and uuid.
- Handle missing values (dropna or fillna) before calculations.
- For any chart: save with a unique uuid filename to 'images/{image_output_dir}', then call plt.close().
- Print ALL results clearly with labels.
- Write as a single executable block — no functions, no classes.
- Do NOT include ```python fences — just the code.
""",

    "general": """
You are an expert Python data analyst. Generate executable pandas code to answer the query.

DataFrame Info:
{df_columns}

Sample Data (first rows):
{df_head}

Rules:
- The DataFrame is already loaded as the variable `df`. Do NOT reload the CSV.
- Use ONLY pandas, matplotlib, seaborn, and uuid — no other libraries.
- Always handle missing values (dropna or fillna) before calculations.
- For any chart: save with a unique uuid filename to the 'images/{image_output_dir}' folder, then call plt.close().
- Print ALL results so they appear in the output.
- Write as a single executable block — no functions, no classes.
- Do NOT include ```python fences or any explanation — just the code.
""",
}

# ─────────────────────────────────────────────
# Domain Report Generation Prompts
# ─────────────────────────────────────────────

DOMAIN_REPORT_PROMPTS: dict[str, str] = {

    "retail": """
You are an expert Retail Data Analyst writing a markdown report.
Use retail terminology: AOV, revenue, margin, category performance, etc.

Dataset columns:
{df_columns}
""",

    "telecom": """
You are an expert Telecom Data Analyst writing a markdown report.
Use telecom terminology: churn rate, ARPU, tenure, contract type, revenue at risk, etc.

Dataset columns:
{df_columns}
""",

    "healthcare": """
You are an expert Healthcare Data Analyst writing a markdown report.
Use healthcare terminology: readmission rate, length of stay, diagnosis, outcomes, etc.
Never include individual patient identifiers.

Dataset columns:
{df_columns}
""",

    "general": """
You are an expert data analyst writing a clear markdown report from analysis results.

Dataset columns available:
{df_columns}
""",
}


# ─────────────────────────────────────────────
# Domain Detection
# Requires at least 2 matching signals to claim a domain.
# Falls back to 'general' when no domain scores 2+.
# ─────────────────────────────────────────────

def detect_domain(column_description: str) -> DomainType:
    col_text = column_description.lower()
    scores: dict[str, int] = {"retail": 0, "telecom": 0, "healthcare": 0}

    for domain, signals in DOMAIN_SIGNALS.items():
        for signal in signals:
            if signal in col_text:
                scores[domain] += 1

    best_domain = max(scores, key=lambda d: scores[d])

    if scores[best_domain] < 2:
        return "general"

    return best_domain  # type: ignore


def get_code_prompt(domain: DomainType) -> str:
    return DOMAIN_CODE_PROMPTS.get(domain, DOMAIN_CODE_PROMPTS["general"])


def get_report_prompt(domain: DomainType) -> str:
    return DOMAIN_REPORT_PROMPTS.get(domain, DOMAIN_REPORT_PROMPTS["general"])
