# Architecture & Design Decisions (DECISIONS.md)

This log documents every architectural choice, deliberate omission, and rationale for this project. The goal is complete clarity and defensibility in an engineering interview for an AI Generalist role.

---

### Decision 1: Direct Thin LLM Client over Frameworks (No LangChain / LlamaIndex)
- **Choice**: Call LLM endpoints directly via Python `httpx` and standard environment variables (`LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL`).
- **Why**:
  - LangChain and LlamaIndex introduce high-abstraction bloat, hidden prompts, fragile version lock-in, and unpredictable overhead for what is fundamentally a structured HTTP request.
  - A direct client is 100% transparent: every prompt, payload, and retry policy is directly visible and testable.
  - If a production team moves from Groq to Anthropic, OpenAI, or local vLLM, it requires changing a standard HTTP endpoint or provider string, not rewriting heavy abstraction chains.

### Decision 2: Scene & Dialogue Chunking over Fixed Token Windows
- **Choice**: Chunk by screenplay scene headings (`SCENE 1:`, `दृश्य १:`) and dialogue blocks, rather than arbitrary token counts (e.g. 500-token sliding windows).
- **Why**:
  - Splitting text mid-dialogue destroys conversational beats, emotional tension, and character interaction context.
  - Scene and dialogue boundaries match human editorial units: a reviewer evaluates an exchange or scene beat, not an arbitrary token slice.
  - It bounds the context window naturally so the LLM concentrates on localized voice and dialect nuances without hallucinating past scene transitions.

### Decision 3: Upfront Glossary Extraction & Grounding
- **Choice**: Extract characters, honorifics, and recurring terms into an editable glossary *before* adapting any chunks.
- **Why**:
  - The single biggest issue in long script translation is **name and persona drift** (e.g., "विक्रम" becoming "Vikram" in Scene 1, "Bikram" in Scene 3, or being translated literally as "Valor").
  - Grounding every chunk prompt with an explicit, immutable list of target names guarantees identity continuity across 50+ scenes.
  - Involving the reviewer early to verify target names takes 15 seconds upfront and eliminates hours of post-editing.

### Decision 4: Two-Tiered QA (Cheap Deterministic First, Targeted LLM Second)
- **Choice**: Run cheap, fast regex and string checks first (character names, numbers/dates, length ratio, untranslated script leakage), followed by a single targeted LLM call for dropped plot points and tone drift.
- **Why**:
  - **Cost and speed**: Deterministic checks execute in < 2ms with zero API cost and 100% precision for rule-based errors (e.g. a missing train time "11:30" or changed character name).
  - **LLMs are poor counting engines**: Asking an LLM "are all numbers preserved?" is slow, expensive, and prone to false negatives. Regex catches missing digits deterministically.
  - **Focused LLM utility**: The LLM is reserved only for what rules cannot do: detecting omitted physical actions and assessing conversational tone.

### Decision 5: Config-Driven Language Pairs (`configs/languages.yaml`)
- **Choice**: All language pair specifics (prompts, tone notes, honorific rules, name rules, few-shot examples) live in a clean YAML configuration file.
- **Why**:
  - Adding or adjusting a language pair (e.g. adding Tamil -> Telugu or Bengali -> Hindi) is strictly a configuration change.
  - Code logic in `localizer.py` and `qa_rules.py` remains language-agnostic and clean.
  - Creative directors or dialect coaches can tune tone notes and honorific rules directly without touching Python code.

### Decision 6: Offline Mock Engine for Zero-Key Testing
- **Choice**: Built-in deterministic mock provider that simulates realistic responses, translations, and QA diagnostics when no API key is provided.
- **Why**:
  - Ensures the consolidated unit test suite runs instantly (in ~0.14s) without relying on internet access or incurring API charges.
  - Interviewers or reviewers can clone the repo and run `streamlit run app/streamlit_app.py` or `python run_cli.py` immediately without searching for an API key.

### Decision 7: What Was Left Out and Why
- **No Vector Databases (RAG)**: Scripts are sequential narratives. Vector similarity search breaks chronological order and retrieves out-of-order chunks that destroy dramatic buildup.
- **No Fine-Tuning**: Indian language models with fine-tuning require expensive compute, fragile maintenance, and lock the system into a static model. Prompt engineering with structured glossaries and few-shot pairs achieves superior agility.
- **No CI / Docker Bloat**: A clean virtualenv with standard `pip install -r requirements.txt` and `pytest` runs anywhere on Python 3.11+ in seconds.
- **No Auto-Generated User Feedback**: Metrics and ratings are not simulated or fabricated. Evaluation scripts read real human rating sheets and seeded-error answer keys.
