"""Pydantic schemas for data structures across the localization pipeline.
"""
from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field


class GlossaryItem(BaseModel):
    """An individual entry in the script glossary."""
    term: str = Field(..., description="Original term/name in the source script")
    category: Literal["character_name", "honorific", "location", "recurring_term"] = Field(
        ..., description="Category of the term"
    )
    target_form: str = Field(..., description="Canonical form in the target language")
    notes: Optional[str] = Field(None, description="Usage guidelines or context")


class Glossary(BaseModel):
    """Container for the script's glossary."""
    items: List[GlossaryItem] = Field(default_factory=list)

    def get_target_name(self, source_name: str) -> Optional[str]:
        """Lookup target form for a given source term (case-insensitive for Latin)."""
        s_clean = source_name.strip().lower()
        for item in self.items:
            if item.term.strip().lower() == s_clean:
                return item.target_form
        return None

    def get_character_names(self) -> Dict[str, str]:
        """Return a mapping of source character name -> target name."""
        return {
            item.term.strip(): item.target_form.strip()
            for item in self.items
            if item.category == "character_name"
        }


class Chunk(BaseModel):
    """A segment of a script (scene, dialogue block, or narrative paragraph)."""
    index: int = Field(..., description="0-based sequence index")
    text: str = Field(..., description="The raw source chunk text")
    chunk_type: str = Field("dialogue_or_scene", description="Type of chunk (scene_heading, dialogue, paragraph)")


class QAFlag(BaseModel):
    """A QA issue flag raised by either deterministic rules or LLM checks."""
    type: Literal[
        "name_drift",
        "number_mismatch",
        "length_anomaly",
        "untranslated_leakage",
        "dropped_plot_point",
        "tone_drift",
        "general_issue"
    ] = Field(..., description="Category of the QA violation")
    severity: Literal["critical", "warning", "info"] = Field(
        ..., description="Severity level for reviewer triage"
    )
    source_excerpt: str = Field(..., description="Snippet from source text that triggered the flag")
    adapted_excerpt: str = Field(..., description="Snippet from adapted text showing the issue")
    short_explanation: str = Field(..., description="Concise explanation for the reviewer")


class QAReport(BaseModel):
    """Aggregation of QA flags for a chunk or full script."""
    flags: List[QAFlag] = Field(default_factory=list)
    passed: bool = Field(True, description="True if no critical flags exist")


class AdaptationResult(BaseModel):
    """The result of adapting a single chunk."""
    chunk_index: int
    source_text: str
    adapted_text: str
    flags: List[QAFlag] = Field(default_factory=list)


class PipelineOutput(BaseModel):
    """Full end-to-end output of the localized script."""
    source_script: str
    language_pair: str
    glossary: Glossary
    adapted_chunks: List[AdaptationResult]
    full_adapted_text: str
    flags: List[QAFlag] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)


class BaselineOutput(BaseModel):
    """Output from the single-prompt baseline run for comparison."""
    source_script: str
    language_pair: str
    adapted_text: str
    stats: Dict[str, Any] = Field(default_factory=dict)
