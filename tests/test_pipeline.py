"""Consolidated test suite for Kuku Localization Engine.
Tests config, chunker, glossary, QA rules, and end-to-end pipeline in one clean file.
"""
import pytest
from src.config import get_pair_config, list_available_pairs
from src.chunker import split_script
from src.schemas import Glossary, GlossaryItem
from src.glossary import extract_glossary_heuristic
from src.qa_rules import check_character_names, check_numbers_and_dates, check_length_ratio
from src.localizer import localize_script
from src.baseline import run_baseline
from src.llm_client import LLMClient


def test_configs_and_language_pairs():
    """Verify all 5 required language pairs exist in config."""
    pairs = list_available_pairs()
    required = ["hindi_to_hinglish", "hindi_to_english", "hindi_to_marathi", "marathi_to_hindi", "english_to_hindi"]
    for r in required:
        assert r in pairs
    cfg = get_pair_config("hindi_to_hinglish")
    assert cfg.source == "hindi" and cfg.target == "hinglish"


def test_chunker_splits_by_scene_and_paragraphs():
    """Verify script chunker correctly handles scenes and blank lines."""
    text = "दृश्य १: कमरा\nविक्रम: मैं जा रहा हूँ।\n\nदृश्य २: स्टेशन\nरोहित: रुको भैया।"
    chunks = split_script(text)
    assert len(chunks) == 2
    assert chunks[0].index == 0 and "दृश्य १" in chunks[0].text
    assert chunks[1].index == 1 and "दृश्य २" in chunks[1].text


def test_glossary_speaker_extraction():
    """Verify heuristic speaker and honorific detection."""
    text = "विक्रम: मैं जा रहा हूँ। पापाजी: मत जाओ।"
    items = extract_glossary_heuristic(text)
    terms = {it.term for it in items}
    assert "विक्रम" in terms
    assert "पापाजी" in terms


def test_qa_deterministic_rules():
    """Verify deterministic QA catches name drift and number mismatch."""
    glossary = Glossary(items=[GlossaryItem(term="विक्रम", category="character_name", target_form="Vikram")])

    # Name drift test
    name_flags = check_character_names("विक्रम: सुनो", "Rahul: Listen", glossary)
    assert len(name_flags) == 1
    assert name_flags[0].type == "name_drift"

    # Number mismatch test (10 omitted)
    num_flags = check_numbers_and_dates("१० लाख रुपये", "Five lakh rupees")
    assert len(num_flags) == 1
    assert num_flags[0].type == "number_mismatch"

    # Length anomaly test (severe truncation)
    len_flags = check_length_ratio("This is a long dramatic sentence spoken by the angry protagonist.", "Wait.")
    assert len(len_flags) == 1
    assert len_flags[0].type == "length_anomaly"


def test_pipeline_and_baseline_end_to_end():
    """Verify pipeline and baseline run with mock client."""
    pair_cfg = get_pair_config("hindi_to_hinglish")
    client = LLMClient(provider="mock")
    script = "विक्रम: पापाजी, मैं आज रात १० बजे जा रहा हूँ।"

    # Pipeline run
    pipe_out = localize_script(script, pair_cfg, llm_client=client)
    assert len(pipe_out.adapted_chunks) == 1
    assert len(pipe_out.full_adapted_text) > 0

    # Baseline run
    base_out = run_baseline(script, pair_cfg, llm_client=client)
    assert len(base_out.adapted_text) > 0
