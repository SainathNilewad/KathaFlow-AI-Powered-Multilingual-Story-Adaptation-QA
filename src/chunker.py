"""Script chunker: Splits scripts into scenes or paragraph beats.
Preserves scene boundaries so the model adapts one dramatic unit at a time.
"""
import re
from typing import List
from src.schemas import Chunk

# Scene headers: "SCENE 1:", "दृश्य १:", "INT.", "EXT."
SCENE_REGEX = re.compile(r"^(?:\s*(?:SCENE|दृश्य|प्रसंग|INT\.|EXT\.).*)$", re.IGNORECASE | re.MULTILINE)


def split_script(text: str) -> List[Chunk]:
    """Split script into scenes if scene headers exist, or by paragraphs (double newlines)."""
    clean = text.strip()
    if not clean:
        return []

    # If script contains scene headings, split around them
    matches = list(SCENE_REGEX.finditer(clean))
    if matches:
        raw_chunks = []
        last_idx = 0
        for m in matches:
            if m.start() > last_idx:
                block = clean[last_idx:m.start()].strip()
                if block:
                    raw_chunks.append(block)
            last_idx = m.start()
        if last_idx < len(clean):
            raw_chunks.append(clean[last_idx:].strip())
    else:
        # Split on double newlines
        raw_chunks = [p.strip() for p in re.split(r"\n\s*\n+", clean) if p.strip()]

    chunks = []
    for i, t in enumerate(raw_chunks):
        c_type = "scene_heading" if SCENE_REGEX.match(t) else "dialogue"
        chunks.append(Chunk(index=i, text=t, chunk_type=c_type))
    return chunks
