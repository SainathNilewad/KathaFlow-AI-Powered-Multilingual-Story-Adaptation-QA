"""Streamlit Web Application: End-to-end Audio Drama Localization & Review Workbench.
Supports pasting/uploading scripts, interactive glossary editing, chunk-by-chunk adaptation,
multi-tiered QA review with flag triage, side-by-side baseline comparison, approval, and local audit logging.
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone
import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import get_pair_config, list_available_pairs
from src.chunker import split_script
from src.glossary import extract_glossary, Glossary, GlossaryItem
from src.localizer import localize_script, adapt_chunk
from src.baseline import run_baseline
from src.qa_rules import run_deterministic_qa
from src.qa_llm import run_llm_qa
from src.llm_client import LLMClient
from src.logger import create_session_id, log_run, log_approval, read_session_logs

# Streamlit Page Config
st.set_page_config(
    page_title="Kuku Script Localizer & QA Workbench",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean audio-drama reviewer aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .flag-critical {
        background-color: #FEF2F2;
        border-left: 4px solid #EF4444;
        padding: 8px 12px;
        margin: 6px 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .flag-warning {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 8px 12px;
        margin: 6px 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .flag-info {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 8px 12px;
        margin: 6px 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .metric-badge {
        font-size: 0.85rem;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = create_session_id()
if "script_text" not in st.session_state:
    st.session_state.script_text = ""
if "glossary" not in st.session_state:
    st.session_state.glossary = None
if "pipeline_output" not in st.session_state:
    st.session_state.pipeline_output = None
if "baseline_output" not in st.session_state:
    st.session_state.baseline_output = None
if "edited_chunks" not in st.session_state:
    st.session_state.edited_chunks = {}
if "approved_state" not in st.session_state:
    st.session_state.approved_state = False

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("🎙️ Kuku Localizer")
st.sidebar.caption("Audio Drama Script Adaptation Workbench")

# 1. Provider & Free LLM Configuration
with st.sidebar.expander("⚡ LLM Provider & Free API Key", expanded=True):
    provider_options = {
        "groq": "Groq (Free: Llama 3.3 70B, Ultra-Fast)",
        "gemini": "Google Gemini (Free: Gemini 1.5 Flash)",
        "openrouter": "OpenRouter (Free :free models)",
        "github": "GitHub Models (Free with GitHub Token)",
        "openai": "OpenAI (gpt-4o-mini)",
        "mock": "Offline Mock Mode (No Key Needed)"
    }
    
    current_provider = os.getenv("LLM_PROVIDER", "groq" if os.getenv("GROQ_API_KEY") else ("gemini" if os.getenv("GEMINI_API_KEY") else "mock")).lower()
    if current_provider not in provider_options:
        current_provider = "mock"
        
    selected_provider = st.selectbox(
        "Provider",
        options=list(provider_options.keys()),
        format_func=lambda x: provider_options[x],
        index=list(provider_options.keys()).index(current_provider)
    )
    
    default_key = os.getenv(f"{selected_provider.upper()}_API_KEY", os.getenv("LLM_API_KEY", ""))
    input_key = st.text_input(
        "API Key",
        value=st.session_state.get("custom_api_key", default_key),
        type="password",
        placeholder="Paste your free key here (gsk_ / AIza...)",
        help="Free keys: Groq -> console.groq.com | Gemini -> aistudio.google.com"
    )
    if input_key:
        st.session_state.custom_api_key = input_key
    
    # Model Override
    custom_model = st.text_input(
        "Model Name (Optional)",
        value=st.session_state.get("custom_model", ""),
        placeholder="Default: provider recommended"
    )
    if custom_model:
        st.session_state.custom_model = custom_model

    # Helpful links for free keys
    st.markdown("""
    <small>
    <b>Get a 100% Free API Key (No Credit Card):</b><br>
    - <a href="https://console.groq.com/keys" target="_blank" style="color:#2563EB;">Groq (Instant, 300+ tok/s)</a><br>
    - <a href="https://aistudio.google.com/app/apikey" target="_blank" style="color:#2563EB;">Google Gemini (DeepMind Indic)</a><br>
    - <a href="https://openrouter.ai/keys" target="_blank" style="color:#2563EB;">OpenRouter (:free models)</a>
    </small>
    """, unsafe_allow_html=True)
    
    test_btn = st.button("🧪 Test API Connection", use_container_width=True)

# Instantiate client dynamically based on user selections
active_key = st.session_state.get("custom_api_key", os.getenv("LLM_API_KEY", ""))
active_model = st.session_state.get("custom_model") or None
client = LLMClient(provider=selected_provider, api_key=active_key, model=active_model)

if test_btn:
    with st.sidebar:
        with st.spinner("Testing API connection..."):
            ok, msg = client.test_connection()
            if ok:
                st.success(msg)
            else:
                st.error(msg)

st.sidebar.markdown("---")

# Language Pair Selector
available_pairs = list_available_pairs()
pair_keys = list(available_pairs.keys())
selected_pair_key = st.sidebar.selectbox(
    "Language Pair",
    options=pair_keys,
    format_func=lambda k: available_pairs[k],
    index=0
)
pair_cfg = get_pair_config(selected_pair_key)

st.sidebar.markdown("---")

# Quick Load Samples
st.sidebar.subheader("Quick Test Samples")
sample_files = {
    "Hindi Drama (Vikram & Rohit)": BASE_DIR / "data/samples/hindi_drama_short.txt",
    "Marathi Family (Tanaji & Sai)": BASE_DIR / "data/samples/marathi_family_short.txt",
    "English Thriller (David & Maya)": BASE_DIR / "data/samples/english_thriller_short.txt",
}

for label, path in sample_files.items():
    if st.sidebar.button(label, use_container_width=True):
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                st.session_state.script_text = f.read()
                st.session_state.pipeline_output = None
                st.session_state.baseline_output = None
                st.session_state.glossary = None
                st.session_state.edited_chunks = {}
                st.session_state.approved_state = False
                st.rerun()

st.sidebar.markdown("---")

# Provider & Mode Indicator
st.sidebar.subheader("System Engine Status")
if client.is_mock():
    st.sidebar.warning("Running in **Mock Mode** (Offline / Zero Key). Responses and QA are deterministically simulated.")
else:
    st.sidebar.success(f"Running Live API: **{client.provider.upper()}**\nModel: `{client.model}`")

enable_llm_qa = st.sidebar.checkbox("Enable LLM QA (Dropped plots & tone)", value=True)

st.sidebar.markdown(f"**Session:** `{st.session_state.session_id}`")


# ==========================================
# MAIN INTERACTION VIEW
# ==========================================
st.markdown('<div class="main-header">Kuku Script Adaptation & Review Workbench</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Preserving character identity, tone, and cultural nuances across Indian audio drama adaptations.</div>', unsafe_allow_html=True)

tabs = st.tabs(["1. Script & Glossary", "2. Review & Edit", "3. Baseline Comparison", "4. Session Audit Logs"])

# ----------------------------------------------------
# TAB 1: SCRIPT & GLOSSARY
# ----------------------------------------------------
with tabs[0]:
    col_input, col_meta = st.columns([2, 1])

    with col_input:
        st.subheader("Source Script")
        uploaded_file = st.file_uploader("Upload plain text script (.txt)", type=["txt"])
        if uploaded_file is not None:
            st.session_state.script_text = uploaded_file.read().decode("utf-8")

        script_area = st.text_area(
            "Paste or edit source script text:",
            value=st.session_state.script_text,
            height=260,
            placeholder="Paste your Hindi, Marathi, or English script here..."
        )
        st.session_state.script_text = script_area

    with col_meta:
        st.subheader("Language Guidelines")
        st.info(f"**Source:** {pair_cfg.source.capitalize()}  ➔  **Target:** {pair_cfg.target.capitalize()}")
        st.markdown(f"**Tone Notes:**\n{pair_cfg.tone_notes}")
        st.markdown(f"**Honorifics:**\n{pair_cfg.honorific_rules}")
        st.markdown(f"**Name Rules:**\n{pair_cfg.name_rules}")

    st.markdown("---")

    # Step 2: Extract Glossary
    st.subheader("Step 2: Script Character & Honorific Glossary")
    st.caption("Characters and honorifics are extracted and anchored upfront so names never drift across long audio episodes.")

    col_btn_gl, col_btn_run = st.columns([1, 1])
    with col_btn_gl:
        if st.button("Extract / Refresh Glossary", type="secondary", use_container_width=True):
            if not st.session_state.script_text.strip():
                st.error("Please paste or upload a script first.")
            else:
                with st.spinner("Extracting dialogue speakers and key terms..."):
                    st.session_state.glossary = extract_glossary(
                        st.session_state.script_text,
                        pair_cfg,
                        llm_client=client
                    )
                st.success("Glossary extracted successfully!")
                st.rerun()

    # Display & edit glossary if available
    if st.session_state.glossary and st.session_state.glossary.items:
        st.markdown("#### Grounded Characters & Terms")
        gl_items = st.session_state.glossary.items

        cols = st.columns(min(3, max(1, len(gl_items))))
        for idx, item in enumerate(gl_items):
            col_target = cols[idx % len(cols)]
            with col_target:
                with st.container(border=True):
                    st.markdown(f"**{item.term}** ({item.category})")
                    new_val = st.text_input(
                        f"Target ({item.term})",
                        value=item.target_form,
                        key=f"gl_item_{idx}"
                    )
                    item.target_form = new_val
                    if item.notes:
                        st.caption(f"Note: {item.notes}")

    # Launch Pipeline Run
    st.markdown("---")
    st.subheader("Step 3: Run Adaptation Pipeline")
    if st.button("Run Structured Adaptation & QA", type="primary", use_container_width=True):
        if not st.session_state.script_text.strip():
            st.error("Please enter a script before running.")
        else:
            prog_bar = st.progress(0, text="Starting localization...")

            def progress_callback(cur, total, msg):
                frac = cur / max(1, total)
                prog_bar.progress(frac, text=msg)

            output = localize_script(
                script_text=st.session_state.script_text,
                pair_config=pair_cfg,
                glossary=st.session_state.glossary,
                enable_llm_qa=enable_llm_qa,
                llm_client=client,
                progress_callback=progress_callback
            )
            prog_bar.empty()

            st.session_state.pipeline_output = output
            st.session_state.glossary = output.glossary

            # Pre-fill editable chunk state
            st.session_state.edited_chunks = {
                r.chunk_index: r.adapted_text for r in output.adapted_chunks
            }
            st.session_state.approved_state = False

            # Log execution
            stats = output.stats
            log_run(
                session_id=st.session_state.session_id,
                language_pair=pair_cfg.display_name,
                source_script=st.session_state.script_text,
                chunks_count=stats.get("chunks_count", 0),
                flags_count=stats.get("flags_count", 0),
                critical_flags=stats.get("critical_flags", 0),
                warning_flags=stats.get("warning_flags", 0),
                info_flags=stats.get("info_flags", 0),
                is_mock=client.is_mock(),
                elapsed_seconds=stats.get("elapsed_seconds", 0.0),
            )

            st.success(f"Pipeline completed in {stats.get('elapsed_seconds')}s across {stats.get('chunks_count')} chunks! Switch to tab '2. Review & Edit'.")


# ----------------------------------------------------
# TAB 2: REVIEW & EDIT (Side-by-side with QA Flags)
# ----------------------------------------------------
with tabs[1]:
    if st.session_state.pipeline_output is None:
        st.info("No adaptation run yet. Please go to tab '1. Script & Glossary' and click 'Run Structured Adaptation & QA'.")
    else:
        pipe_out = st.session_state.pipeline_output
        stats = pipe_out.stats

        # Metrics banner
        m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
        m_c1.metric("Total Chunks", stats.get("chunks_count", 0))
        m_c2.metric("Execution Time", f"{stats.get('elapsed_seconds', 0)}s")
        m_c3.metric("Critical Flags", stats.get("critical_flags", 0), delta_color="inverse")
        m_c4.metric("Warning Flags", stats.get("warning_flags", 0), delta_color="inverse")
        m_c5.metric("Review Status", "Approved" if st.session_state.approved_state else "Pending Review")

        st.markdown("---")

        # Severity Filter
        severity_filter = st.radio(
            "Filter Flags:",
            options=["All", "Critical Only", "Warnings & Critical", "Clean Chunks Only"],
            horizontal=True
        )

        st.subheader("Side-by-Side Reviewer Workspace")
        st.caption("Review source and adapted chunks side-by-side. Inspect QA flags and edit the adapted text directly.")

        for r in pipe_out.adapted_chunks:
            # Check filter
            has_crit = any(f.severity == "critical" for f in r.flags)
            has_warn = any(f.severity == "warning" for f in r.flags)
            has_any = len(r.flags) > 0

            if severity_filter == "Critical Only" and not has_crit:
                continue
            if severity_filter == "Warnings & Critical" and not (has_crit or has_warn):
                continue
            if severity_filter == "Clean Chunks Only" and has_any:
                continue

            with st.container(border=True):
                head_col1, head_col2 = st.columns([1, 1])
                head_col1.markdown(f"**Chunk #{r.chunk_index + 1}** ({r.chunk_index})")
                if r.flags:
                    badges = " ".join([
                        f"🔴 {flg.type}" if flg.severity == "critical" else f"🟠 {flg.type}"
                        for flg in r.flags
                    ])
                    head_col2.markdown(f"<div style='text-align: right;'>{badges}</div>", unsafe_allow_html=True)
                else:
                    head_col2.markdown("<div style='text-align: right; color: green;'>🟢 Passed Clean</div>", unsafe_allow_html=True)

                c_src, c_tgt = st.columns([1, 1])
                with c_src:
                    st.text_area(
                        "Source Excerpt",
                        value=r.source_text,
                        height=120,
                        disabled=True,
                        key=f"src_area_{r.chunk_index}"
                    )

                with c_tgt:
                    current_val = st.session_state.edited_chunks.get(r.chunk_index, r.adapted_text)
                    edited_text = st.text_area(
                        "Adapted Text (Editable)",
                        value=current_val,
                        height=120,
                        key=f"tgt_area_{r.chunk_index}"
                    )
                    st.session_state.edited_chunks[r.chunk_index] = edited_text

                # Display Flags attached to this chunk
                if r.flags:
                    st.markdown("**QA Diagnostics:**")
                    for flg in r.flags:
                        css_class = f"flag-{flg.severity}"
                        st.markdown(
                            f"""
                            <div class="{css_class}">
                                <b>[{flg.severity.upper()}] {flg.type.replace('_', ' ').title()}:</b> {flg.short_explanation}<br/>
                                <small>Source snippet: <i>"{flg.source_excerpt}"</i> &nbsp;|&nbsp; Target: <i>"{flg.adapted_excerpt}"</i></small>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

        # Assemble full final script
        final_script_chunks = [
            st.session_state.edited_chunks.get(i, pipe_out.adapted_chunks[i].adapted_text)
            for i in range(len(pipe_out.adapted_chunks))
        ]
        final_full_script = "\n\n".join(final_script_chunks)

        st.markdown("---")
        st.subheader("Step 6: Reviewer Sign-off & Export")

        col_app, col_down_txt, col_down_json = st.columns(3)

        with col_app:
            if st.button("Approve Script", type="primary", use_container_width=True):
                st.session_state.approved_state = True
                log_approval(
                    session_id=st.session_state.session_id,
                    edits_count=sum(
                        1 for i, txt in st.session_state.edited_chunks.items()
                        if txt != pipe_out.adapted_chunks[i].adapted_text
                    ),
                    remaining_flags_count=len(pipe_out.flags),
                    download_format="approved_txt",
                )
                st.success("Script approved! Session recorded in audit log.")

        with col_down_txt:
            st.download_button(
                label="Download Adapted Script (.txt)",
                data=final_full_script,
                file_name=f"kuku_adapted_{selected_pair_key}_{st.session_state.session_id[:8]}.txt",
                mime="text/plain",
                use_container_width=True
            )

        with col_down_json:
            export_payload = {
                "session_id": st.session_state.session_id,
                "language_pair": pair_cfg.display_name,
                "approved": st.session_state.approved_state,
                "full_adapted_script": final_full_script,
                "flags": [f.model_dump() for f in pipe_out.flags],
                "stats": pipe_out.stats
            }
            st.download_button(
                label="Download QA Report (.json)",
                data=json.dumps(export_payload, indent=2, ensure_ascii=False),
                file_name=f"kuku_qa_report_{selected_pair_key}_{st.session_state.session_id[:8]}.json",
                mime="application/json",
                use_container_width=True
            )


# ----------------------------------------------------
# TAB 3: BASELINE COMPARISON
# ----------------------------------------------------
with tabs[2]:
    st.subheader("Pipeline vs. Plain-Prompt Baseline Comparison")
    st.caption("A single naive prompt on the entire script serves as the baseline to evaluate where the chunked & grounded pipeline wins.")

    if not st.session_state.script_text.strip():
        st.info("Please load or paste a script in Tab 1 first.")
    else:
        if st.button("Run Baseline Comparison", type="secondary"):
            with st.spinner("Running single-prompt baseline on whole script..."):
                st.session_state.baseline_output = run_baseline(
                    script_text=st.session_state.script_text,
                    pair_config=pair_cfg,
                    llm_client=client
                )
            st.success("Baseline completed!")

        if st.session_state.baseline_output is not None:
            base_out = st.session_state.baseline_output

            b_col1, b_col2 = st.columns(2)
            with b_col1:
                st.markdown("### Structured Pipeline Output")
                if st.session_state.pipeline_output:
                    p_text = "\n\n".join(
                        st.session_state.edited_chunks.get(i, r.adapted_text)
                        for i, r in enumerate(st.session_state.pipeline_output.adapted_chunks)
                    )
                    st.text_area("Pipeline Script", value=p_text, height=350, disabled=True)
                    st.caption(f"Words: {len(p_text.split())} | Chunks: {len(st.session_state.pipeline_output.adapted_chunks)} | QA Flags: {len(st.session_state.pipeline_output.flags)}")
                else:
                    st.warning("Run the pipeline in Tab 1 to compare here.")

            with b_col2:
                st.markdown("### Naive Plain Baseline Output")
                st.text_area("Baseline Script", value=base_out.adapted_text, height=350, disabled=True)
                st.caption(f"Words: {base_out.stats.get('adapted_word_count')} | Single un-chunked prompt | QA Flags: Not Available")

            st.markdown("---")
            st.markdown("#### Key Comparative Advantages of Structured Pipeline:")
            st.markdown("""
            1. **No Name Drift**: Glossary grounding prevents proper names from drifting or being translated literally.
            2. **Granular Review & Editing**: Reviewers can approve chunk-by-chunk rather than rereading an entire script.
            3. **Automated Diagnostics**: Deterministic rules catch missing numbers, missing dates, and language leaks instantly.
            4. **Audit Trail**: Every change and flag is tracked for quality control and team calibration.
            """)


# ----------------------------------------------------
# TAB 4: SESSION AUDIT LOGS
# ----------------------------------------------------
with tabs[3]:
    st.subheader("Local Session Audit Logs")
    st.caption("Real, unedited logs recorded locally to logs/session_logs.jsonl.")

    entries = read_session_logs()
    if not entries:
        st.info("No session logs recorded yet. Run a localization or approve a script to record real usage.")
    else:
        st.write(f"Total Logged Events: **{len(entries)}**")

        # Summary statistics from real logs
        runs = [e for e in entries if e.get("event_type") == "pipeline_run"]
        approvals = [e for e in entries if e.get("event_type") == "reviewer_approval"]

        l_c1, l_c2, l_c3 = st.columns(3)
        l_c1.metric("Pipeline Runs", len(runs))
        l_c2.metric("Script Approvals", len(approvals))
        total_flags_logged = sum(r.get("flags_count", 0) for r in runs)
        l_c3.metric("Total Flags Caught", total_flags_logged)

        st.markdown("---")
        st.markdown("#### Recent Log Entries (JSON Lines)")
        for entry in reversed(entries[-15:]):
            with st.expander(f"{entry.get('timestamp')} - {entry.get('event_type')}"):
                st.json(entry)
