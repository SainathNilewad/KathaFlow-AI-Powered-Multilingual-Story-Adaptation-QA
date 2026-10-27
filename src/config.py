"""Configuration loader for languages and QA rules.
Adding or modifying languages requires changes in configs/languages.yaml only.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
from pydantic import BaseModel, Field

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = BASE_DIR / "configs"


class LanguageDef(BaseModel):
    name: str
    script: str
    code: str


class FewShotExample(BaseModel):
    source: str
    adapted: str


class LanguagePairConfig(BaseModel):
    source: str
    target: str
    display_name: str
    tone_notes: str
    honorific_rules: str
    name_rules: str
    few_shot_examples: List[FewShotExample] = Field(default_factory=list)


def load_languages_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and parse configs/languages.yaml."""
    path = config_path or (CONFIGS_DIR / "languages.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Languages config file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_qa_rules_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and parse configs/qa_rules.yaml."""
    path = config_path or (CONFIGS_DIR / "qa_rules.yaml")
    if not path.exists():
        raise FileNotFoundError(f"QA rules config file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_available_pairs() -> Dict[str, str]:
    """Returns mapping of pair_key -> display_name."""
    raw = load_languages_config()
    pairs = raw.get("pairs", {})
    return {k: v.get("display_name", k) for k, v in pairs.items()}


def get_pair_config(pair_key: str) -> LanguagePairConfig:
    """Retrieve typed LanguagePairConfig for a given key (e.g. 'hindi_to_hinglish')."""
    raw = load_languages_config()
    pairs = raw.get("pairs", {})
    if pair_key not in pairs:
        available = ", ".join(pairs.keys())
        raise KeyError(f"Unknown language pair '{pair_key}'. Available pairs: {available}")
    return LanguagePairConfig(**pairs[pair_key])


def get_qa_config() -> Dict[str, Any]:
    """Retrieve full QA configuration dictionary."""
    return load_qa_rules_config()
