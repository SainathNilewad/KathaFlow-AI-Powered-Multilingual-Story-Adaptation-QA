"""Localizer: Adapts scripts chunk-by-chunk using glossary grounding and runs QA.
"""
import time
from typing import Optional, Callable, List
from src.schemas import Chunk, Glossary, AdaptationResult, PipelineOutput, QAFlag
from src.chunker import split_script
from src.glossary import extract_glossary, format_glossary_for_prompt
from src.llm_client import LLMClient
from src.config import LanguagePairConfig
from src.qa_rules import run_deterministic_qa
from src.qa_llm import run_llm_qa


def adapt_chunk(
    chunk: Chunk,
    pair_config: LanguagePairConfig,
    glossary: Glossary,
    llm_client: Optional[LLMClient] = None,
) -> str:
    """Adapt a single script chunk into target language using glossary constraints."""
    client = llm_client or LLMClient()
    glossary_str = format_glossary_for_prompt(glossary)

    prompt = f"""Adapt this audio drama script chunk from {pair_config.source} to {pair_config.target}.
TONE: {pair_config.tone_notes}
HONORIFIC RULES: {pair_config.honorific_rules}
STRICT GLOSSARY:
{glossary_str}

Return ONLY the adapted dialogue/action without conversational preambles.
CHUNK TO ADAPT:
{chunk.text}
"""
    adapted = client.generate(prompt=prompt, temperature=0.1).strip()
    if adapted.startswith("```") and adapted.endswith("```"):
        lines = adapted.split("\n")
        if len(lines) >= 3:
            adapted = "\n".join(lines[1:-1]).strip()
    return adapted


def localize_script(
    script_text: str,
    pair_config: LanguagePairConfig,
    glossary: Optional[Glossary] = None,
    enable_llm_qa: bool = True,
    llm_client: Optional[LLMClient] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> PipelineOutput:
    """Run full pipeline: chunk -> extract glossary -> adapt chunks -> run QA checks."""
    t0 = time.time()
    client = llm_client or LLMClient()
    chunks = split_script(script_text)

    if not chunks:
        return PipelineOutput(
            source_script=script_text,
            language_pair=pair_config.display_name,
            glossary=glossary or Glossary(),
            adapted_chunks=[],
            full_adapted_text="",
            flags=[]
        )

    if glossary is None:
        glossary = extract_glossary(script_text, pair_config, client)

    adapted_results: List[AdaptationResult] = []
    all_flags: List[QAFlag] = []

    for i, c in enumerate(chunks):
        if progress_callback:
            progress_callback(i + 1, len(chunks), f"Adapting chunk {i+1}/{len(chunks)}...")

        adapted_text = adapt_chunk(c, pair_config, glossary, client)

        # Tier 1: Deterministic QA
        c_flags = run_deterministic_qa(c.text, adapted_text, glossary, pair_config.target)

        # Tier 2: LLM QA
        if enable_llm_qa:
            c_flags.extend(run_llm_qa(c.text, adapted_text, pair_config, client))

        all_flags.extend(c_flags)
        adapted_results.append(AdaptationResult(
            chunk_index=i, source_text=c.text, adapted_text=adapted_text, flags=c_flags
        ))

    full_adapted = "\n\n".join(r.adapted_text for r in adapted_results)
    elapsed = round(time.time() - t0, 2)

    return PipelineOutput(
        source_script=script_text,
        language_pair=pair_config.display_name,
        glossary=glossary,
        adapted_chunks=adapted_results,
        full_adapted_text=full_adapted,
        flags=all_flags,
        stats={
            "chunks_count": len(chunks),
            "source_word_count": len(script_text.split()),
            "adapted_word_count": len(full_adapted.split()),
            "flags_count": len(all_flags),
            "critical_flags": sum(1 for f in all_flags if f.severity == "critical"),
            "warning_flags": sum(1 for f in all_flags if f.severity == "warning"),
            "elapsed_seconds": elapsed,
            "is_mock": client.is_mock()
        }
    )
