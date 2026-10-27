"""Thin LLM client: Calls free and commercial LLM APIs directly via standard HTTP.
Supported Free Providers:
- Groq (Free: Llama 3.3 70B, ultra-fast 300+ tok/s, free key at console.groq.com)
- Google Gemini (Free tier: Gemini 1.5 Flash, free key at aistudio.google.com)
- OpenRouter (Free models ending with :free, key at openrouter.ai)
- GitHub Models (Free with GitHub Personal Access Token at models.inference.ai.azure.com)
Falls back to a deterministic offline mock when no API key is provided.
"""
import os
import json
import time
import httpx
from typing import Optional, Dict, Any, Tuple
from dotenv import load_dotenv

load_dotenv()

PROVIDER_URLS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "github": "https://models.inference.ai.azure.com/chat/completions",
    "openai": "https://api.openai.com/v1/chat/completions",
}

DEFAULT_FREE_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-1.5-flash",
    "openrouter": "meta-llama/llama-3.3-70b-instruct:free",
    "github": "gpt-4o-mini",
    "openai": "gpt-4o-mini",
}


class LLMClient:
    """Direct, framework-free client for free and commercial LLM endpoints."""

    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None, model: Optional[str] = None):
        # 1. Resolve API key from arguments or environment
        self.api_key = (
            api_key
            or os.getenv("LLM_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("GITHUB_TOKEN")
            or os.getenv("OPENAI_API_KEY")
            or ""
        ).strip()

        # 2. Resolve provider
        if provider:
            self.provider = provider.lower()
        elif os.getenv("LLM_PROVIDER"):
            self.provider = os.getenv("LLM_PROVIDER").lower()
        elif os.getenv("GROQ_API_KEY"):
            self.provider = "groq"
        elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            self.provider = "gemini"
        elif os.getenv("OPENROUTER_API_KEY"):
            self.provider = "openrouter"
        elif os.getenv("GITHUB_TOKEN"):
            self.provider = "github"
        elif os.getenv("OPENAI_API_KEY"):
            self.provider = "openai"
        elif self.api_key:
            # Default to groq if key looks like gsk_
            if self.api_key.startswith("gsk_"):
                self.provider = "groq"
            elif self.api_key.startswith("AIza"):
                self.provider = "gemini"
            elif self.api_key.startswith("ghp_"):
                self.provider = "github"
            else:
                self.provider = "groq"
        else:
            self.provider = "mock"

        # 3. Resolve model
        default_model = DEFAULT_FREE_MODELS.get(self.provider, "llama-3.3-70b-versatile")
        self.model = model or os.getenv("LLM_MODEL") or default_model
        self.base_url = os.getenv("LLM_BASE_URL", "")

        if not self.api_key:
            self.provider = "mock"

    def is_mock(self) -> bool:
        return self.provider == "mock" or not bool(self.api_key)

    def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.1, json_mode: bool = False) -> str:
        """Send prompt to LLM endpoint or return deterministic mock response."""
        if self.is_mock() or not self.api_key:
            return self._mock_response(prompt, json_mode)

        # Standard OpenAI-compatible completion endpoint
        url = self.base_url or PROVIDER_URLS.get(self.provider, "https://api.groq.com/openai/v1/chat/completions")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # OpenRouter specific headers for attribution
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://kuku.fm"
            headers["X-Title"] = "Kuku Script Localizer"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        with httpx.Client(timeout=45.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    def test_connection(self) -> Tuple[bool, str]:
        """Test API connectivity with a minimal 1-token call."""
        if self.is_mock():
            return True, "Running in Offline Mock Mode (no external API calls required)."
        try:
            res = self.generate("Say hello in one word", temperature=0.0)
            return True, f"Success! Model {self.model} on {self.provider.upper()} responded: '{res.strip()}'"
        except Exception as e:
            return False, f"Connection error on {self.provider.upper()}: {str(e)}"

    def _mock_response(self, prompt: str, json_mode: bool) -> str:
        """Deterministic, realistic mock for offline testing without external API keys."""
        p_lower = prompt.lower()

        # Mock Glossary Extraction
        if json_mode and ("glossar" in p_lower or "extract characters" in p_lower):
            if "david" in p_lower or "maya" in p_lower:
                return json.dumps({
                    "items": [
                        {"term": "David", "category": "character_name", "target_form": "डेविड", "notes": "Detective"},
                        {"term": "Maya", "category": "character_name", "target_form": "माया", "notes": "Partner"}
                    ]
                })
            elif "tanaji" in p_lower or "तानाजी" in p_lower:
                return json.dumps({
                    "items": [
                        {"term": "तानाजी", "category": "character_name", "target_form": "तानाजी", "notes": "Father"},
                        {"term": "सई", "category": "character_name", "target_form": "सई", "notes": "Daughter"}
                    ]
                })
            else:
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

        # Extract the actual script dialogue from the prompt
        chunk_text = prompt
        if "CHUNK TO ADAPT:" in prompt:
            chunk_text = prompt.split("CHUNK TO ADAPT:", 1)[1].strip()
        elif "SCRIPT TO ADAPT:" in prompt:
            chunk_text = prompt.split("SCRIPT TO ADAPT:", 1)[1].strip()

        # Detect source and target language
        is_to_hindi = "to hindi" in p_lower
        is_to_hinglish = "to hinglish" in p_lower
        is_to_marathi = "to marathi" in p_lower
        is_to_english = "to english" in p_lower

        adapted_lines = []
        for line in chunk_text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Preserve scene headers
            if line.upper().startswith("SCENE") or line.startswith("दृश्य") or line.startswith("प्रसंग"):
                adapted_lines.append(line)
                continue

            # English -> Hindi adaptation
            if is_to_hindi:
                trans = (
                    line.replace("David:", "डेविड:")
                    .replace("Maya:", "माया:")
                    .replace("Officer:", "ऑफिसर:")
                    .replace("Listen to me", "मेरी बात सुनो")
                    .replace("listen to me", "मेरी बात सुनो")
                    .replace("We don't have much time", "हमारे पास ज़्यादा वक़्त नहीं है")
                    .replace("Wait!", "रुको!")
                    .replace("Look at this", "यह देखो")
                    .replace("Where are you going?", "तुम कहाँ जा रही हो?")
                    .replace("10 minutes", "१० मिनट")
                    .replace("10:00", "१०:००")
                    .replace("24 hours", "२४ घंटे")
                )
                if ":" in line and ":" not in trans:
                    spk, dia = line.split(":", 1)
                    trans = f"{spk}: {dia.strip()}"
                elif ":" in trans and trans == line:
                    spk, dia = line.split(":", 1)
                    trans = f"{spk}: {dia.strip()} (अनुवाद)"
                adapted_lines.append(trans)

            # Hindi -> Hinglish adaptation
            elif is_to_hinglish:
                trans = (
                    line.replace("विक्रम:", "Vikram:")
                    .replace("रोहित:", "Rohit:")
                    .replace("पापाजी:", "Papaji:")
                    .replace("दादी:", "Dadi:")
                    .replace("अगर आज तू इस घर से बाहर गया", "Agar aaj tu is ghar se bahar gaya")
                    .replace("तो वापस आने की उम्मीद मत रखना", "toh wapas aane ki umeed mat rakhna")
                    .replace("मैंने ये फैसला सोच-समझकर लिया है", "maine yeh faisla soch-samajhkar liya hai")
                    .replace("भैया, ज़रा सुनिए!", "Bhaiya, zara suniye!")
                    .replace("१० लाख रुपये", "10 lakh rupees")
                    .replace("११:३०", "11:30")
                    .replace("२४ घंटे", "24 hours")
                )
                adapted_lines.append(trans)

            # Hindi/English -> Marathi adaptation
            elif is_to_marathi:
                trans = (
                    line.replace("Vikram:", "विक्रम:")
                    .replace("Rohit:", "रोहित:")
                    .replace("Papaji:", "बाबा:")
                    .replace("Tanaji:", "तानाजी:")
                    .replace("Sai:", "सई:")
                    .replace("नहीं", "नाही")
                    .replace("हैं", "आहेत")
                    .replace("था", "होता")
                    .replace("लेकिन", "पण")
                    .replace("१० लाख रुपये", "१० लाख रुपये")
                    .replace("२४ घंटे", "२४ तास")
                )
                adapted_lines.append(trans)

            # Other -> English adaptation
            elif is_to_english:
                trans = (
                    line.replace("विक्रम:", "Vikram:")
                    .replace("रोहित:", "Rohit:")
                    .replace("पापाजी:", "Papaji:")
                    .replace("तानाजी:", "Tanaji:")
                    .replace("सई:", "Sai:")
                )
                adapted_lines.append(trans)

            else:
                adapted_lines.append(line)

        return "\n".join(adapted_lines) if adapted_lines else chunk_text
