"""Glossary extractor: Identifies characters and honorifics upfront
so names and respect terms stay consistent across all scenes.
"""
import re
import json
from typing import Optional, List, Set
from src.schemas import Glossary, GlossaryItem
from src.llm_client import LLMClient
from src.config import LanguagePairConfig

# Matches character names preceding colons (e.g. "VIKRAM:" or "विक्रम:")
SPEAKER_REGEX = re.compile(
    r"(?:^|\n|[।\.\?!]\s+)\s*([A-Z\u0900-\u097F][A-Za-z0-9\.\u0900-\u097F]*(?:\s+[A-Z\u0900-\u097F][A-Za-z0-9\.\u0900-\u097F]*){0,3})\s*:(?!\d|[०-९])"
)

# Common cultural honorifics
COMMON_HONORIFICS = ["पापाजी", "पिताजी", "बाबूजी", "भैया", "दीदी", "चाचाजी", "भाभी", "दादा", "ताई", "काका", "साहेब"]
SCENE_KEYWORDS = re.compile(r"^(?:SCENE|दृश्य|प्रसंग|EXT\.|INT\.)\b", re.IGNORECASE)


def extract_glossary_heuristic(script_text: str, pair_config: Optional[LanguagePairConfig] = None) -> List[GlossaryItem]:
    """Fast regex speaker and honorific detection."""
    items: List[GlossaryItem] = []
    seen: Set[str] = set()

    for spk in set(SPEAKER_REGEX.findall(script_text)):
        spk_clean = spk.strip()
        if not spk_clean or SCENE_KEYWORDS.search(spk_clean) or spk_clean in seen:
            continue
        seen.add(spk_clean)
        # Default target name
        target_name = spk_clean
        if pair_config and pair_config.target in ("hinglish", "english"):
            name_map = {"विक्रम": "Vikram", "रोहित": "Rohit", "नेहा": "Neha", "तानाजी": "Tanaji", "सई": "Sai"}
            target_name = name_map.get(spk_clean, spk_clean)
        items.append(GlossaryItem(term=spk_clean, category="character_name", target_form=target_name))

    for h in COMMON_HONORIFICS:
        if h in script_text and h not in seen:
            seen.add(h)
            items.append(GlossaryItem(term=h, category="honorific", target_form=h, notes="Respect marker"))

    return items


def extract_glossary(script_text: str, pair_config: LanguagePairConfig, llm_client: Optional[LLMClient] = None) -> Glossary:
    """Extract glossary using LLM or regex fallback."""
    client = llm_client or LLMClient()
    items = extract_glossary_heuristic(script_text, pair_config)

    # If running with real LLM, ask LLM to enrich the glossary
    if not client.is_mock():
        prompt = f"Extract characters, honorifics, and locations from this script into JSON items:\n{script_text[:2000]}"
        try:
            resp = client.generate(prompt=prompt, system_prompt="Output JSON: {'items': [{'term': ..., 'category': ..., 'target_form': ...}]}", json_mode=True)
            llm_items = json.loads(resp).get("items", [])
            for e in llm_items:
                if e.get("term") not in {it.term for it in items}:
                    items.append(GlossaryItem(**e))
        except Exception:
            pass

    return Glossary(items=items)


def format_glossary_for_prompt(glossary: Glossary) -> str:
    """Format glossary as clear prompt instructions."""
    if not glossary.items:
        return "No specific glossary terms."
    return "\n".join(f"- **{it.term}** -> **{it.target_form}** [{it.category}]" for it in glossary.items)
