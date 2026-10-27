"""Local session logger: Persists real pipeline runs and reviewer decisions to JSONL.
Provides verifiable usage logs for interviewing and empirical analysis.
"""
import os
import json
import uuid
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOGS_DIR / "session_logs.jsonl"


def _ensure_log_dir():
    """Ensure the logs directory exists."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def create_session_id() -> str:
    """Generate a clean unique session ID."""
    return f"sess_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def log_event(event_type: str, data: Dict[str, Any]):
    """Append a structured JSON line entry to logs/session_logs.jsonl."""
    _ensure_log_dir()
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        **data,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_run(
    session_id: str,
    language_pair: str,
    source_script: str,
    chunks_count: int,
    flags_count: int,
    critical_flags: int,
    warning_flags: int,
    info_flags: int,
    is_mock: bool,
    elapsed_seconds: float,
):
    """Log the completion of a localization run."""
    script_hash = hashlib.sha256(source_script.encode("utf-8")).hexdigest()[:10]
    log_event(
        "pipeline_run",
        {
            "session_id": session_id,
            "language_pair": language_pair,
            "script_hash": script_hash,
            "source_chars": len(source_script),
            "source_words": len(source_script.split()),
            "chunks_count": chunks_count,
            "flags_count": flags_count,
            "critical_flags": critical_flags,
            "warning_flags": warning_flags,
            "info_flags": info_flags,
            "is_mock": is_mock,
            "elapsed_seconds": elapsed_seconds,
        }
    )


def log_approval(
    session_id: str,
    edits_count: int,
    remaining_flags_count: int,
    download_format: str,
):
    """Log human reviewer approval and download action."""
    log_event(
        "reviewer_approval",
        {
            "session_id": session_id,
            "edits_count": edits_count,
            "remaining_flags_count": remaining_flags_count,
            "download_format": download_format,
        }
    )


def read_session_logs() -> List[Dict[str, Any]]:
    """Read all persisted log entries from session_logs.jsonl."""
    if not LOG_FILE.exists():
        return []
    entries = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                try:
                    entries.append(json.loads(line_str))
                except json.JSONDecodeError:
                    continue
    return entries
