# 🤖 AutoAnalyst — Agentic Data Analysis Agent

AutoAnalyst is a powerful, agentic AI designed to transform raw CSV data into actionable insights. It uses a **LangGraph** workflow to write, sanitize, and execute Python code, generating markdown reports and data visualizations automatically.

---

## ✨ Key Features

-   **Intelligent Code Generation**: Writes precise pandas and matplotlib code based on your natural language questions.
-   **Multi-Layer Security**: Every script is checked by both static analysis and an LLM security reviewer before execution.
-   **Session Memory**: Remembers previous questions so you can ask follow-up questions (e.g., "Now show that by region").
-   **Organized Assets**: Automatically creates unique, timestamped folders for every analysis session to keep your charts organized.
-   **Email Integration**: Send your final analysis reports directly to your email address.
-   **Beautiful UI**: A premium Streamlit interface for a seamless user experience.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+
- A Groq API Key (get one at [console.groq.com](https://console.groq.com))

### 2. Installation
1. Clone the repository.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Configuration
Create a `.env` file in the root directory (or copy `.env.example`) and fill in your details:
```ini
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Email (Optional - for Email Report feature)
SMTP_HOST=smtp.mail.yahoo.com
SMTP_PORT=587
SMTP_USER=your-email@yahoo.com
SMTP_PASSWORD=your-app-password
SMTP_FROM=AutoAnalyst <your-email@yahoo.com>
```

---

## 🛠️ How to Run

You need to run **two** processes concurrently:

### Step 1: Start the Backend (FastAPI)
```bash
uvicorn src.main:app --reload
```
The API will be available at `http://localhost:8000`.

### Step 2: Start the Frontend (Streamlit)
```bash
streamlit run streamlit_app.py
```
The UI will open in your browser at `http://localhost:8501`.

---

## 📖 How to Use

1.  **Upload**: Drag and drop your CSV file into the sidebar.
2.  **Describe**: Click "Get Column Descriptions" to see what data the agent can analyze.
3.  **Analyze**: Type a question in the chat (e.g., "What is the average total bill by day of the week?").
4.  **Visualize**: The agent will automatically generate charts if relevant to your question.
5.  **Email**: Enter your email address at the bottom and click "Send Report" to get a copy of the analysis.

---

## 📁 Project Structure

-   `src/`: Core application logic (FastAPI, LangGraph nodes, LLM client).
-   `tests/`: Unit and integration tests.
-   `docs/`: Detailed API and Architecture documentation.
-   `images/`: Generated charts (organized by session).
-   `uploads/`: Temporary storage for uploaded CSVs.

---

## 🛡️ Security
This agent executes generated Python code in a restricted sandbox. It uses `PythonAstREPLTool` with limited locals and performs a two-layer security check to prevent dangerous operations like file deletion or network calls.
