"""Thin LLM client: Calls OpenAI/Groq/Gemini directly via HTTP.
Falls back to a simple mock when no API key is provided, so tests and demos run offline.
"""
import os
import json
import httpx
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """Minimal direct LLM client with zero framework dependencies."""

    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None, model: Optional[str] = None):
        # Read credentials from environment
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY") or ""
        self.provider = (provider or os.getenv("LLM_PROVIDER") or ("openai" if self.api_key else "mock")).lower()
        self.model = model or os.getenv("LLM_MODEL") or ("gpt-4o-mini" if self.provider == "openai" else "llama-3.3-70b-versatile")
        self.base_url = os.getenv("LLM_BASE_URL", "")

        if not self.api_key:
            self.provider = "mock"

    def is_mock(self) -> bool:
        return self.provider == "mock"

    def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.1, json_mode: bool = False) -> str:
        """Send prompt to LLM provider or return simple mock response."""
        if self.is_mock():
            return self._mock_response(prompt, json_mode)

        # Standard OpenAI / Groq compatible endpoint
        url = self.base_url or ("https://api.groq.com/openai/v1/chat/completions" if self.provider == "groq" else "https://api.openai.com/v1/chat/completions")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    def _mock_response(self, prompt: str, json_mode: bool) -> str:
        """Simple deterministic mock for offline testing without API keys."""
        p_lower = prompt.lower()

        # Mock Glossary Extraction
        if json_mode and ("glossar" in p_lower or "extract characters" in p_lower):
            return json.dumps({
                "items": [
                    {"term": "विक्रम", "category": "character_name", "target_form": "Vikram", "notes": "Protagonist"},
                    {"term": "रोहित", "category": "character_name", "target_form": "Rohit", "notes": "Brother"},
                    {"term": "पापाजी", "category": "honorific", "target_form": "Papaji", "notes": "Respect marker"}
                ]
            })

        # Mock QA check
        if json_mode:
            return json.dumps({"flags": []})

        # Mock Chunk Adaptation or Baseline: simple clean mock translation
        lines = []
        for line in prompt.split("\n"):
            if any(term in line for term in ["विक्रम", "रोहित", "पापाजी", "दादी"]):
                # Simple romanization for demonstration
                cleaned = line.replace("विक्रम", "Vikram").replace("रोहित", "Rohit").replace("पापाजी", "Papaji")
                lines.append(cleaned)
        return "\n".join(lines) if lines else f"[Adapted]: {prompt[:120]}..."
