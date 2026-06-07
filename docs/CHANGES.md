# AutoAnalyst — Improvement Notes

This document explains every bug, hallucination source, and optimization
found in the original code, and how each one was fixed. Written to be
understandable for a first-time project.

---

## Part 1 — Hallucination: Why It Happens & How We Fixed It

### What is hallucination?

In AI, "hallucination" means the model invents information that isn't
real. In your project, this shows up as the report containing numbers
that were NEVER in the actual data — the LLM made them up.

### Root causes found (3 causes, all fixed)

---

#### Cause 1 — The LLM's "temperature" was too high for report writing

**File:** `llm_client.py` → `generate_report()`

**The problem:**
Temperature controls how "creative" the LLM is. At temperature 0, the
model picks the most likely next word every time — predictable, faithful.
At higher temperatures it "explores" alternatives — which in a report
means rounding 47.3% to "approximately 50%" or inventing a figure it
thinks sounds right.

The original code used `_get_llm()` (which defaults to temperature=0)
but this was not explicitly enforced at the report step.

**The fix:**
```python
# BEFORE
chain = prompt | _get_llm() | StrOutputParser()

# AFTER — explicitly enforce temperature=0 for report generation
chain = prompt | _get_llm(temperature=0) | StrOutputParser()
```

---

#### Cause 2 — The execution output wasn't clearly marked as the authority

**File:** `llm_client.py` → `REPORT_GENERATION_USER` prompt

**The problem:**
The original prompt said "Every number must come from the Analysis Output
above" — but the output block wasn't visually separated from the rest of
the instructions. The LLM treated it as one big wall of text and still
sometimes used numbers from its training memory instead of the actual
output.

**The fix — use a clear delimiter:**
```
══════════════════════════════════════════
AUTHORITATIVE DATA (copy numbers from here — do NOT invent any)
══════════════════════════════════════════
{execution_results}
══════════════════════════════════════════
```

The box makes it obvious what the source of truth is. The prompt also
now explicitly says: "If a figure is not in the block, write 'not
available' — never guess."

---

#### Cause 3 — Empty execution_results was being passed to the code prompt

**File:** `domain_prompts.py` — all `DOMAIN_CODE_PROMPTS`

**The problem:**
The code generation prompts included a `{execution_results}` placeholder.
But at code-GENERATION time, there are no results yet (we haven't run
the code). So it was being filled with an empty string:

```python
# In generate_python_code(), results don't exist yet
code = call_llm_with_retry(chain, {
    ...
    "execution_results": "",   # ← always empty here!
})
```

This is confusing to the LLM — it sees "here are your results: [nothing]"
and sometimes uses that absence as permission to make up its own results.

**The fix:**
`{execution_results}` was removed entirely from code-generation prompts.
Code prompts generate code. Report prompts consume results. They are
separate steps and should have separate prompts.

---

## Part 2 — Other Bugs Fixed

---

### Bug 1 — False positive error detection

**File:** `llm_client.py` → `execute_python_code()`

**The problem:**
```python
# ORIGINAL — too broad
if results and "error" in results.lower():
    return {"execution_error": results, ...}
```

If your dataset had a column called `error_rate` or `mean_absolute_error`,
the printed output would contain the word "error", and the agent would
think the code FAILED — even though it ran perfectly. This caused
unnecessary retries and sometimes a dead end.

**The fix — check for actual Python tracebacks:**
```python
# AFTER — only trigger on real Python errors
if results and (
    "Traceback (most recent call last)" in results
    or re.search(r"\b\w+Error:", results)
):
    return {"execution_error": results, ...}
```

A real Python error always starts with "Traceback" or ends with
"SomeError: message". A legitimate result that mentions "error" does not.

---

### Bug 2 — Markdown fences breaking code execution

**File:** `llm_client.py` → `generate_python_code()` and `re_generate_python_code()`

**The problem:**
LLMs often wrap their output in markdown code fences:
```python
```python
import pandas as pd
result = df.groupby('region').mean()
```
```

When you try to run this as Python code, you get a `SyntaxError` because
` ```python ` and ` ``` ` are not valid Python.

**The fix — strip fences before using the code:**
```python
def _strip_markdown_fences(code: str) -> str:
    code = re.sub(r"^```(?:python)?\s*\n?", "", code.strip(), flags=re.IGNORECASE)
    code = re.sub(r"\n?```\s*$", "", code.strip())
    return code.strip()
```

This runs on every piece of generated code before it enters the
sanitizer or executor.

---

### Bug 3 — DataFrame read from disk on every retry

**File:** `llm_client.py` → `generate_python_code()`

**The problem:**
```python
# ORIGINAL — re-reads from disk every time
def generate_python_code(state):
    df = pd.read_csv(state["csv_file_path"])
```

The DataFrame is stored in `state["data_frame"]` after the first read.
But the code ignored it and read from disk again every time — including
on every retry. For a 50 MB file, this is slow and wasteful.

**The fix:**
```python
# AFTER — re-use what's already in state
df = state.get("data_frame")
if df is None:
    df = pd.read_csv(state["csv_file_path"])
```

---

### Bug 4 — re_generate_python_code passed wrong variables to chain.invoke()

**File:** `llm_client.py` → `re_generate_python_code()`

**The problem:**
```python
# ORIGINAL
new_code = call_llm_with_retry(chain, {
    "image_output_dir": state["image_output_dir"]   # ← not used by the prompt!
})
```

The `CODE_FIX_SYSTEM` prompt is formatted with `.format()` before being
passed to the LLM — so `image_output_dir` is already baked in. Passing
it again to `chain.invoke()` meant LangChain was looking for an
unformatted `{image_output_dir}` in the prompt that no longer existed,
which raised a silent error in some LangChain versions.

**The fix:**
```python
# AFTER — pass empty dict; all values already injected via .format()
new_code = call_llm_with_retry(chain, {})
```

---

### Bug 5 — Guardrail blocked legitimate chart-saving code

**File:** `guardrails.py` → `DANGEROUS_CODE_PATTERNS`

**The problem:**
```python
# ORIGINAL — too broad
r"\bopen\s*\(.*['\"]w['\"]",   # blocks ALL file writes
```

This regex blocked any `open(..., 'w')` call. But matplotlib's `savefig()`
internally opens files for writing. So legitimately saving a chart to
`images/abc.png` could trigger this guardrail.

**The fix — only block writes OUTSIDE the images/ directory:**
```python
# AFTER — allow images/ writes, block everything else
r"open\s*\(\s*['\"](?!images/)[^'\"]+['\"]\s*,\s*['\"]w['\"]",
```

The `(?!images/)` part is a "negative lookahead" — it means "match this
pattern UNLESS the filename starts with images/".

---

### Bug 6 — Conversation history was in wrong order

**File:** `memory.py` → `get_context()`

**The problem:**
The original code walked turns in reverse (newest first) for budget
control, then used `parts.insert(1, entry)` to put each entry after the
header. But since we inserted at position 1 each time, the last inserted
entry (which was the oldest one we collected) ended up at position 1 —
meaning the final order was still newest-to-oldest, the opposite of
what you want in a conversation.

**The fix:**
```python
# Collect in reverse order (newest first, for budget control)
collected_entries = []
for turn in reversed(session.turns):
    ...
    collected_entries.append(entry)

# Reverse again so output is chronological (oldest first)
collected_entries.reverse()
return header + "\n" + "\n".join(collected_entries)
```

---

## Part 3 — Performance Optimizations

---

### Optimization 1 — Domain detection cached in state

**File:** `llm_client.py`

The `detect_domain()` function scans all column names every time it's
called. In the original code it was called in both `generate_python_code`
and `generate_report`. The fix stores the result in `state["domain"]`
so `generate_report` just reads it:

```python
# generate_python_code now stores it:
return {"Python_Code": code, "data_frame": df, "domain": domain}

# generate_report now reads it:
domain = state.get("domain", "general")   # no second scan
```

---

### Optimization 2 — No redundant disk reads

Already covered in Bug 3. Summary: reading a CSV from disk is slow.
We do it once, store it in state, and reuse it everywhere.

---

## Part 4 — Quick Reference: Which File Changed and Why

| File | What changed |
|---|---|
| `llm_client.py` | Hallucination fixes (temp, prompt, fence stripping, false-positive error, df caching, invoke() fix) |
| `domain_prompts.py` | Removed `{execution_results}` from code prompts; cleaner report prompt wording |
| `guardrails.py` | Fixed open() regex; added check_execution_output(); strip fences before scanning |
| `memory.py` | Fixed conversation order bug; added get_session() helper |

`streamlit_app.py`, `main.py`, `models.py`, and `email_utils.py` had
no bugs — they are unchanged.
