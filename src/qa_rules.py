"""Deterministic QA rules: Fast, cheap checks for names, numbers, length, and language leakage.
"""
import re
from typing import List, Optional, Dict
from src.schemas import QAFlag, Glossary

DEV_DIGIT_MAP = str.maketrans("०१२३४५६७८९", "0123456789")


def check_character_names(source_text: str, adapted_text: str, glossary: Glossary) -> List[QAFlag]:
    """Flag if a character in source is missing its expected target name in adaptation."""
    flags = []
    char_map = glossary.get_character_names()
    adapted_lower = adapted_text.lower()

    for src_name, tgt_name in char_map.items():
        if re.search(rf"\b{re.escape(src_name)}\b", source_text, re.IGNORECASE):
            if not re.search(rf"\b{re.escape(tgt_name.strip())}\b", adapted_lower, re.IGNORECASE):
                flags.append(QAFlag(
                    type="name_drift",
                    severity="critical",
                    source_excerpt=src_name,
                    adapted_excerpt=adapted_text[:100],
                    short_explanation=f"Character '{src_name}' was in source, but target form '{tgt_name}' is missing."
                ))
    return flags


def check_numbers_and_dates(source_text: str, adapted_text: str) -> List[QAFlag]:
    """Flag if numbers or dates in source are omitted from adaptation."""
    flags = []
    src_norm = source_text.translate(DEV_DIGIT_MAP)
    tgt_norm = adapted_text.translate(DEV_DIGIT_MAP).lower()

    numbers = set(re.findall(r"\b[0-9]+\b", src_norm))
    for num in numbers:
        if not re.search(rf"\b{re.escape(num)}\b", tgt_norm):
            flags.append(QAFlag(
                type="number_mismatch",
                severity="critical",
                source_excerpt=f"Number {num}",
                adapted_excerpt=adapted_text[:100],
                short_explanation=f"Number '{num}' from source was omitted or altered."
            ))
    return flags


def check_length_ratio(source_text: str, adapted_text: str, min_ratio: float = 0.45, max_ratio: float = 2.20) -> List[QAFlag]:
    """Flag if chunk word count deviates suspiciously (truncation or hallucination)."""
    flags = []
    src_words = len(source_text.split())
    tgt_words = len(adapted_text.split())

    if src_words >= 6:
        ratio = tgt_words / max(1, src_words)
        if ratio < min_ratio:
            flags.append(QAFlag(
                type="length_anomaly",
                severity="warning",
                source_excerpt=source_text[:80],
                adapted_excerpt=adapted_text[:80],
                short_explanation=f"Adapted text is noticeably short ({ratio:.2f}x source words). Possible omission."
            ))
        elif ratio > max_ratio:
            flags.append(QAFlag(
                type="length_anomaly",
                severity="warning",
                source_excerpt=source_text[:80],
                adapted_excerpt=adapted_text[:80],
                short_explanation=f"Adapted text is suspiciously long ({ratio:.2f}x source words). Possible hallucination."
            ))
    return flags


def check_untranslated_leakage(source_text: str, adapted_text: str, target_lang: str) -> List[QAFlag]:
    """Flag if untranslated source script leaked into the target output."""
    flags = []
    if target_lang.lower() in ("english", "hinglish"):
        leaks = re.findall(r"[\u0900-\u097F]{5,}", adapted_text)
        if leaks:
            flags.append(QAFlag(
                type="untranslated_leakage",
                severity="warning",
                source_excerpt=source_text[:80],
                adapted_excerpt=leaks[0],
                short_explanation=f"Found unadapted Devanagari text '{leaks[0]}' in {target_lang} output."
            ))
    return flags


def run_deterministic_qa(source_text: str, adapted_text: str, glossary: Glossary, target_lang: str = "hinglish") -> List[QAFlag]:
    """Run all 4 deterministic checks in sequence."""
    flags = []
    flags.extend(check_character_names(source_text, adapted_text, glossary))
    flags.extend(check_numbers_and_dates(source_text, adapted_text))
    flags.extend(check_length_ratio(source_text, adapted_text))
    flags.extend(check_untranslated_leakage(source_text, adapted_text, target_lang))
    return flags
