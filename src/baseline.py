"""Baseline model: Single plain prompt on the whole script without chunking,
glossary grounding, or step-by-step QA.
Used as the benchmark to measure where the structured pipeline beats naive prompting.
"""
import time
import logging
from typing import Optional
from src.schemas import BaselineOutput
from src.llm_client import LLMClient
from src.config import LanguagePairConfig

logger = logging.getLogger("kuku.baseline")


def run_baseline(
    script_text: str,
    pair_config: LanguagePairConfig,
    llm_client: Optional[LLMClient] = None,
) -> BaselineOutput:
    """Run a single plain prompt on the whole script.

    Why this baseline:
    A direct, un-chunked translation prompt is the default naive solution.
    It typically suffers from:
    1. Character name drift across long passages
    2. Dropped lines / truncated micro-beats in long outputs
    3. Cultural honorific dilution
    4. Zero automated QA verification or flag generation
    """
    start_time = time.time()
    client = llm_client or LLMClient()

    prompt = f"""Translate and adapt the following {pair_config.source} audio/micro-drama script into {pair_config.target}.
Maintain the natural flow and tone of the characters.
Return ONLY the translated script without preambles or notes.

SCRIPT TO TRANSLATE:
```
{script_text}
```
"""

    adapted = client.generate(
        prompt=prompt,
        system_prompt=f"You translate {pair_config.source} scripts into {pair_config.target}.",
        temperature=0.2
    ).strip()

    if adapted.startswith("```") and adapted.endswith("```"):
        lines = adapted.split("\n")
        if len(lines) >= 3:
            adapted = "\n".join(lines[1:-1]).strip()

    elapsed = time.time() - start_time
    stats = {
        "source_word_count": len(script_text.split()),
        "adapted_word_count": len(adapted.split()),
        "elapsed_seconds": round(elapsed, 2),
        "is_mock": client.is_mock()
    }

    return BaselineOutput(
        source_script=script_text,
        language_pair=pair_config.display_name,
        adapted_text=adapted,
        stats=stats
    )
