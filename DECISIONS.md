# Architecture & Design Decisions (DECISIONS.md)

This log documents every architectural choice, deliberate omission, and technical rationale for the **Script Localization & Review Workbench**. The goal is complete clarity and defensibility in an engineering interview for an **AI Generalist** role (Kuku AI Labs).

---

### Decision 1: Notebook-First Implementation over Complex Web/App Packages
- **Choice**: Consolidate the entire end-to-end pipeline into `localization_workbench.ipynb` in 12 clearly numbered, sequentially executable sections. The Streamlit web app and scattered multi-file library structure (`src/`, `app/`, `eval/`) were safely archived.
- **Rationale**:
  - In an interview setting, having code dispersed across 15+ files forces the interviewer to switch contexts constantly and makes it harder to defend how data flows between components.
  - A notebook allows an engineer to inspect intermediate states directly: chunk boundaries, extracted Pydantic models, raw LLM prompts, deterministic flag outputs, and evaluation metrics.
  - It demonstrates an essential AI Generalist virtue: **knowing when a simple solution is enough** rather than over-engineering infrastructure for a task that is primarily analytical and iterative.

---

### Decision 2: Direct Thin LLM Client over Heavy Frameworks (No LangChain / LlamaIndex)
- **Choice**: Call LLM endpoints directly via standard HTTP requests (`httpx`), reading credentials and models from environment variables.
- **Rationale**:
  - LangChain and LlamaIndex introduce high abstraction overhead, hidden system prompts, brittle dependency trees, and breaking API changes.
  - A thin direct client is 100% transparent: the prompt, system instructions, temperature, and JSON schema payload are explicitly visible in one cell.
  - If a production team moves from OpenAI to Groq, Anthropic, or an on-prem vLLM instance, it requires updating a standard HTTP base URL, not re-architecting deep framework pipelines.

---

### Decision 3: Upfront Glossary Extraction with Phonetic Alias Spelling Tolerances
- **Choice**: Extract characters, honorifics, and key terms into a validated Pydantic glossary *before* adapting any scenes, explicitly supporting phonetic alias lists (e.g. `canonical_term: "विक्रम"`, `preferred_target: "Vikram"`, `aliases: ["Bikram", "Vicky"]`).
- **Rationale**:
  - The single most common failure in audio drama localization is **name drift** across episodes (e.g. *विक्रम* translated as *Valor*, or varying between *Vikram* and *Bikram* across scenes).
  - In conversational Hinglish and Indian language subtitles, names have **no standardized Latin spelling** (e.g., *Rohan/Rohin*, *Bikram/Vikram*, *Kabir/Kabeer*).
  - Storing approved spelling variations as aliases prevents the deterministic QA engine from generating false-positive alarms on legitimate phonetic transcriptions.
  - Reviewers can inspect and approve the glossary in 15 seconds upfront, anchoring the LLM's chunk-by-chunk adaptation with immutable entity groundings.

---

### Decision 4: Scene & Dialogue Chunking over Fixed Token Windows
- **Choice**: Chunk scripts by screenplay scene headings (`SCENE 1:`, `दृश्य १:`) and dialogue boundaries, preserving stable chunk IDs (`<script_id>_scene_001`), rather than slicing text by arbitrary token counts (e.g. 500-token sliding windows).
- **Rationale**:
  - Splitting text mid-dialogue destroys conversational rhythm, dramatic tension, and relational context between characters.
  - Screenplay boundaries match how human script editors and voice actors work.
  - Providing bounded context (the immediate preceding scene) preserves conversational continuity across scene cuts without resending the entire script, keeping latency and token costs minimal.

---

### Decision 5: Two-Tiered QA (Deterministic Heuristics First, Targeted LLM Second)
- **Choice**: Run ultra-fast, zero-API-cost deterministic checks first (name consistency, number/date preservation, length ratio, script leakage), followed by a single targeted LLM call for semantic checks (dropped plot points, tone drift).
- **Rationale**:
  - **Cost and speed**: Deterministic checks execute in < 2ms with zero API cost and 100% precision for rule-based discrepancies (e.g. missing a 24-hour deadline or a 10 lakh rupee transaction).
  - **LLMs are poor counting engines**: Asking an LLM "are all numbers preserved?" is slow, expensive, and prone to hallucinations. Regex catches missing digits deterministically.
  - **Clear separation of concerns**: LLMs are reserved exclusively for subjective linguistic nuances: conversational tone and dramatic beat continuity.

---

### Decision 6: Indic Script Word Boundaries & Same-Script Heuristics
- **Choice**: Use Devanagari token extraction (`re.findall(r"[\u0900-\u097F]+", text)`) for word matching instead of regex `\b`, and document same-script leakage checks as weak heuristics.
- **Technical Nuance**:
  - In Python's standard `re` engine, Unicode combining marks (such as the Devanagari Anusvara `\u0902` in `"नहीं"`) are classified as non-spacing marks (`Mn`), not `\w`. Consequently, standard regex `\b` fails to recognize word boundaries on Indic words ending in combining characters.
  - For cross-script pairs (e.g. Hindi to English/Hinglish), script leakage is easily caught via Unicode block checks (`[\u0900-\u097F]{4,}`).
  - For same-script pairs (e.g. Hindi to Marathi, both Devanagari), character script checks cannot work. We use a configurable wordlist of language-specific markers (e.g., Hindi `"नहीं"`, `"लेकिन"`, `"अगर"` vs. Marathi `"नाही"`, `"पण"`, `"जर"`).
  - **Explicit limitation**: Same-script wordlist checks are heuristic only; they can miss morphological inflections and should never be treated as definitive proof of translation error.

---

### Decision 7: File-Based Human Review (HTML + CSV) without Interactive Widgets
- **Choice**: Export reviews as `review_sheet.html` (for side-by-side visual inspection with highlighted flags) and `review_decisions.csv` (for editing decisions outside Jupyter).
- **Rationale**:
  - Libraries like `ipywidgets` are notoriously fragile across different Jupyter distributions, notebook extensions, and headless environments.
  - Real human reviewers (script editors, voice directors) typically do not use Jupyter Notebooks. They work in Excel, Google Sheets, or web tables.
  - CSV export allows reviewers to work asynchronously in their preferred spreadsheet tool. The notebook then reads the completed CSV back, strictly preserving reviewer edits and approvals without overwriting unreviewed chunks.

---

### Decision 8: Token Usage Tracking and Avoiding Fabricated Cost Metrics
- **Choice**: Log raw `input_tokens`, `output_tokens`, and runtime to `logs/session_logs.jsonl` per run; never hardcode dollar costs or invent adoption statistics.
- **Rationale**:
  - Model pricing changes constantly across providers (OpenAI, Groq, Google, Anthropic). Hardcoding prices in code produces stale or misleading financial metrics.
  - Logging exact token counts allows financial calculations to be computed dynamically at runtime based on current vendor contracts.
  - Fabricating synthetic benchmarks or user engagement metrics undermines engineering integrity.

---

### Decision 9: What Was Deliberately Left Out and Why
- **No Vector Databases (RAG)**: Screenplays and audio dramas are strictly sequential, linear narratives. Vector similarity retrieval fetches out-of-order scenes, destroying dramatic pacing and plot continuity.
- **No Fine-Tuning**: Fine-tuning proprietary or open-source models for multi-dialect Indian localization requires expensive compute, creates rigid model locks, and cannot easily incorporate real-time editorial glossary edits. Prompt engineering grounded by structured glossaries and few-shot examples is far more agile.
- **No Web UI Frameworks in V1**: Prioritizing core AI reliability, evaluation rigor, and deterministic validation over frontend widget code.
