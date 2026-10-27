"""LLM-based QA evaluator: Catches subtle nuances that deterministic rules miss.
Specifically focuses on:
1. Dropped plot points, actions, or character intentions
2. Tone drift from target drama style (e.g. overly polite, stilted, or losing emotional beats)
"""
import json
import logging
from typing import List, Optional
from src.schemas import QAFlag
from src.llm_client import LLMClient
from src.config import LanguagePairConfig

logger = logging.getLogger("kuku.qa_llm")


def run_llm_qa(
    source_text: str,
    adapted_text: str,
    pair_config: LanguagePairConfig,
    llm_client: Optional[LLMClient] = None,
) -> List[QAFlag]:
    """Execute a targeted LLM QA evaluation to catch dropped plot points and tone drift.
    Validates all output flags with Pydantic QAFlag.
    """
    client = llm_client or LLMClient()

    prompt = f"""You are an expert audio-drama script supervisor and reviewer.
Compare the following SOURCE script excerpt with its ADAPTED version.

Source Language: {pair_config.source}
Target Language: {pair_config.target}

Target Tone Guidelines:
{pair_config.tone_notes}

Your task is to detect ONLY two types of problems that automated regex cannot catch:
1. 'dropped_plot_point' (severity: 'critical' or 'warning'): An action, crucial statement, plot reveal, or character motivation present in the source was omitted or altered in meaning.
2. 'tone_drift' (severity: 'warning' or 'info'): The adaptation drastically altered the character's emotional intensity, social register, or used overly academic/stilted phrasing that ruins the audio drama experience.

If there are NO dropped plot points and NO significant tone drift, return an empty list: {{"flags": []}}.

Output ONLY valid JSON:
{{
  "flags": [
    {{
      "type": "dropped_plot_point" | "tone_drift",
      "severity": "critical" | "warning" | "info",
      "source_excerpt": "Exact short quote from source where issue originates",
      "adapted_excerpt": "Exact short quote from adapted text, or '[Omitted in adaptation]'",
      "short_explanation": "One clear sentence explaining the specific issue for the human reviewer"
    }}
  ]
}}

SOURCE TEXT:
```
{source_text}
```

ADAPTED TEXT:
```
{adapted_text}
```
"""

    flags: List[QAFlag] = []
    try:
        raw_resp = client.generate(
            prompt=prompt,
            system_prompt="You evaluate script localization accuracy and tone. Output JSON only.",
            temperature=0.0,
            json_mode=True
        )
        data = json.loads(raw_resp)
        raw_flags = data.get("flags", [])
        for f in raw_flags:
            try:
                # Validate flag type strictly
                f_type = f.get("type")
                if f_type not in ("dropped_plot_point", "tone_drift", "general_issue"):
                    f["type"] = "general_issue"
                flags.append(QAFlag(**f))
            except Exception as e:
                logger.debug(f"Skipping invalid QA flag from LLM: {e}")
    except Exception as e:
        logger.warning(f"LLM QA evaluation call failed: {e}")

    return flags
