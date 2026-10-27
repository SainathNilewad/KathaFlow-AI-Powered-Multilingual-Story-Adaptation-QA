"""QA Evaluation Scorer: Evaluates the QA engine against human-seeded ground truth errors.
Reports Recall (detection rate of known errors) and False Alarm count per script.
Adheres strictly to the rule: Do not auto-label answer keys; reads the user's answer key.
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_pair_config
from src.schemas import Glossary, GlossaryItem
from src.qa_rules import run_deterministic_qa
from src.qa_llm import run_llm_qa
from src.llm_client import LLMClient


def score_qa_against_key(answer_key_path: Path, skip_llm: bool = False):
    if not answer_key_path.exists():
        print(f"Error: Answer key file not found at '{answer_key_path}'", file=sys.stderr)
        sys.exit(1)

    with open(answer_key_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    source_text = data.get("source_text", "").strip()
    adapted_text = data.get("adapted_text", "").strip()
    source_file_path = data.get("source_file")
    adapted_file_path = data.get("adapted_file")

    if not source_text and source_file_path:
        with open(source_file_path, "r", encoding="utf-8") as f:
            source_text = f.read().strip()

    if not adapted_text and adapted_file_path:
        with open(adapted_file_path, "r", encoding="utf-8") as f:
            adapted_text = f.read().strip()

    pair_key = data.get("language_pair", "hindi_to_hinglish")
    pair_cfg = get_pair_config(pair_key)
    known_errors: List[Dict[str, Any]] = data.get("known_errors", [])

    # Reconstruct glossary if provided in key
    glossary_items = [GlossaryItem(**item) for item in data.get("glossary_items", [])]
    glossary = Glossary(items=glossary_items)

    print("=" * 65)
    print(" QA BENCHMARK EVALUATION (SEEDED ERROR TEST)")
    print("=" * 65)
    print(f"Answer Key File   : {answer_key_path.name}")
    print(f"Language Pair     : {pair_cfg.display_name}")
    print(f"Total Seeded Errors : {len(known_errors)}")

    if not known_errors:
        print("\n[!] The answer key has 0 known errors defined.")
        print("Please define seeded errors in the answer key file using the template format.")
        print("=" * 65)
        return

    client = LLMClient()

    # 1. Run deterministic checks
    detected_flags = run_deterministic_qa(
        source_text=source_text,
        adapted_text=adapted_text,
        glossary=glossary,
        target_lang=pair_cfg.target
    )

    # 2. Run LLM QA check if not disabled
    if not skip_llm:
        llm_flags = run_llm_qa(
            source_text=source_text,
            adapted_text=adapted_text,
            pair_config=pair_cfg,
            llm_client=client
        )
        detected_flags.extend(llm_flags)

    print(f"Total Flags Raised: {len(detected_flags)}")

    # Match detected flags against known errors
    matched_error_ids = set()
    matched_flag_indices = set()

    for err in known_errors:
        err_id = err.get("id", "unknown")
        expected_type = err.get("type", "").lower()
        err_snippet = err.get("target_snippet", "").lower()

        for f_idx, flg in enumerate(detected_flags):
            if f_idx in matched_flag_indices:
                continue

            type_match = (
                not expected_type
                or flg.type.lower() == expected_type
                or (expected_type in ("dropped_plot_point", "length_anomaly") and flg.type in ("dropped_plot_point", "length_anomaly"))
            )

            snippet_match = (
                not err_snippet
                or err_snippet in flg.adapted_excerpt.lower()
                or err_snippet in flg.source_excerpt.lower()
                or err_snippet in flg.short_explanation.lower()
            )

            if type_match and snippet_match:
                matched_error_ids.add(err_id)
                matched_flag_indices.add(f_idx)
                break

    # Calculate metrics
    total_seeded = len(known_errors)
    caught_count = len(matched_error_ids)
    missed_count = total_seeded - caught_count
    false_alarms = len(detected_flags) - len(matched_flag_indices)

    recall = (caught_count / total_seeded) * 100 if total_seeded > 0 else 0.0
    precision = (caught_count / len(detected_flags)) * 100 if detected_flags else 0.0

    print("\nPERFORMANCE METRICS:")
    print(f"  - Known Errors Caught (TP) : {caught_count} / {total_seeded} ({recall:.1f}% Recall)")
    print(f"  - Known Errors Missed (FN) : {missed_count}")
    print(f"  - False Alarms (FP)        : {false_alarms}")
    if detected_flags:
        print(f"  - Precision                : {precision:.1f}%")

    print("\nSEEDED ERROR BREAKDOWN:")
    for err in known_errors:
        eid = err.get("id")
        status = "[CAUGHT]" if eid in matched_error_ids else "[MISSED]"
        print(f"  {status} {eid} ({err.get('type')}): {err.get('description')}")

    if false_alarms > 0:
        print("\nUNMATCHED FLAGS (Potential False Alarms or Unlabeled Flaws):")
        for f_idx, flg in enumerate(detected_flags):
            if f_idx not in matched_flag_indices:
                print(f"  - [{flg.severity.upper()}] ({flg.type}): {flg.short_explanation}")

    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Score QA engine against a seeded-error answer key.")
    parser.add_argument("--key", "-k", required=True, help="Path to answer key JSON file")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM check and test only deterministic rules")
    args = parser.parse_args()

    score_qa_against_key(Path(args.key), skip_llm=args.skip_llm)


if __name__ == "__main__":
    main()
