"""Evaluation scoring: Reads human-filled blind rating sheet and calculates
empirical win/tie/loss rates and average ratings for Pipeline vs. Baseline.
Adheres strictly to the rule: Never invent ratings. Calculates metrics only from real human entries.
"""
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def score_ratings(sheet_path: Path, key_path: Path):
    if not sheet_path.exists():
        print(f"Error: Rating sheet not found at '{sheet_path}'", file=sys.stderr)
        sys.exit(1)
    if not key_path.exists():
        print(f"Error: Rating key not found at '{key_path}'", file=sys.stderr)
        sys.exit(1)

    with open(key_path, "r", encoding="utf-8") as f:
        key_data = json.load(f)
    samples_key = key_data.get("samples", {})

    pipeline_wins = 0
    baseline_wins = 0
    ties = 0

    pipeline_scores: List[float] = []
    baseline_scores: List[float] = []
    unrated_samples: List[str] = []

    rows: List[Dict[str, str]] = []
    with open(sheet_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    for r in rows:
        sample_id = r.get("sample_id", "").strip()
        pref_raw = r.get("preferred (A/B/Tie)", "").strip().upper()
        score_a_raw = r.get("score_A (1-5)", "").strip()
        score_b_raw = r.get("score_B (1-5)", "").strip()

        if sample_id not in samples_key:
            continue

        mapping = samples_key[sample_id]
        system_a = mapping["variant_A"]
        system_b = mapping["variant_B"]

        # Parse numerical scores if provided
        score_a = float(score_a_raw) if score_a_raw.replace(".", "", 1).isdigit() else None
        score_b = float(score_b_raw) if score_b_raw.replace(".", "", 1).isdigit() else None

        if score_a is not None and score_b is not None:
            if system_a == "pipeline":
                pipeline_scores.append(score_a)
                baseline_scores.append(score_b)
            else:
                baseline_scores.append(score_a)
                pipeline_scores.append(score_b)

        # Parse preference (or deduce from score if preference was omitted)
        if pref_raw in ("A", "VARIANT_A", "1"):
            preferred_system = system_a
        elif pref_raw in ("B", "VARIANT_B", "2"):
            preferred_system = system_b
        elif pref_raw in ("TIE", "EQUAL", "T"):
            preferred_system = "tie"
        elif score_a is not None and score_b is not None:
            if score_a > score_b:
                preferred_system = system_a
            elif score_b > score_a:
                preferred_system = system_b
            else:
                preferred_system = "tie"
        else:
            unrated_samples.append(sample_id)
            continue

        if preferred_system == "pipeline":
            pipeline_wins += 1
        elif preferred_system == "baseline":
            baseline_wins += 1
        else:
            ties += 1

    total_evaluated = pipeline_wins + baseline_wins + ties

    print("=" * 65)
    print(" HUMAN EVALUATION SCORING REPORT")
    print("=" * 65)
    print(f"Rating Sheet File : {sheet_path.name}")
    print(f"Total Rows in CSV : {len(rows)}")
    print(f"Rated Samples (N) : {total_evaluated}")
    print(f"Unrated Samples   : {len(unrated_samples)}")

    if total_evaluated == 0:
        print("\n[!] No rated samples found in this sheet.")
        print("Please open the CSV file and rate at least one sample in 'preferred (A/B/Tie)'.")
        print("=" * 65)
        return

    pipe_win_pct = (pipeline_wins / total_evaluated) * 100
    base_win_pct = (baseline_wins / total_evaluated) * 100
    tie_pct = (ties / total_evaluated) * 100

    print("\nHEAD-TO-HEAD PREFERENCE (Win / Tie / Loss):")
    print(f"  - Pipeline Wins : {pipeline_wins:3d} ({pipe_win_pct:5.1f}%)")
    print(f"  - Baseline Wins : {baseline_wins:3d} ({base_win_pct:5.1f}%)")
    print(f"  - Ties          : {ties:3d} ({tie_pct:5.1f}%)")

    if pipeline_scores and baseline_scores:
        avg_pipe = sum(pipeline_scores) / len(pipeline_scores)
        avg_base = sum(baseline_scores) / len(baseline_scores)
        diff = avg_pipe - avg_base
        print("\nAVERAGE SCORES (1 to 5 scale):")
        print(f"  - Pipeline Avg Score : {avg_pipe:.2f} / 5.0 (n={len(pipeline_scores)})")
        print(f"  - Baseline Avg Score : {avg_base:.2f} / 5.0 (n={len(baseline_scores)})")
        print(f"  - Score Difference   : {diff:+.2f}")

    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Score a completed human blind rating sheet.")
    parser.add_argument("--sheet", "-s", required=True, help="Path to filled rating CSV")
    parser.add_argument("--key", "-k", required=True, help="Path to rating key JSON")
    args = parser.parse_args()

    score_ratings(Path(args.sheet), Path(args.key))


if __name__ == "__main__":
    main()
