"""Evaluation tooling: Generates a blind side-by-side CSV comparing Baseline vs. Pipeline.
Randomizes assignment of Variant A and Variant B so the human reviewer cannot bias results.
Saves a decoupled key in eval/results/ for automated scoring.
"""
import sys
import csv
import json
import random
from pathlib import Path
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_pair_config, list_available_pairs
from src.localizer import localize_script
from src.baseline import run_baseline
from src.llm_client import LLMClient


def create_rating_sheet(
    sample_files: list[Path],
    pair_key: str = "hindi_to_hinglish",
    output_dir: Path = Path("data/ratings"),
    key_dir: Path = Path("eval/results"),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sheet_path = output_dir / f"blind_rating_sheet_{pair_key}_{timestamp}.csv"
    key_path = key_dir / f"rating_key_{pair_key}_{timestamp}.json"

    pair_cfg = get_pair_config(pair_key)
    client = LLMClient()

    rows = []
    key_mapping = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "language_pair": pair_key,
            "display_name": pair_cfg.display_name,
            "llm_provider": client.provider,
            "is_mock": client.is_mock(),
        },
        "samples": {}
    }

    print(f"Generating blind rating sheet for {len(sample_files)} sample(s)...")

    for idx, sample_file in enumerate(sample_files, 1):
        sample_id = f"sample_{idx:03d}_{sample_file.stem}"
        with open(sample_file, "r", encoding="utf-8") as f:
            source_text = f.read().strip()

        # Run both systems
        print(f"  [{idx}/{len(sample_files)}] Running pipeline & baseline on {sample_file.name}...")
        pipeline_out = localize_script(source_text, pair_cfg, llm_client=client)
        baseline_out = run_baseline(source_text, pair_cfg, llm_client=client)

        pipe_text = pipeline_out.full_adapted_text
        base_text = baseline_out.adapted_text

        # Randomize Variant A and Variant B
        is_pipeline_a = random.choice([True, False])
        if is_pipeline_a:
            variant_a = pipe_text
            variant_b = base_text
            mapping = {"variant_A": "pipeline", "variant_B": "baseline"}
        else:
            variant_a = base_text
            variant_b = pipe_text
            mapping = {"variant_A": "baseline", "variant_B": "pipeline"}

        key_mapping["samples"][sample_id] = mapping

        rows.append({
            "sample_id": sample_id,
            "language_pair": pair_cfg.display_name,
            "source_script": source_text,
            "variant_A": variant_a,
            "variant_B": variant_b,
            "preferred (A/B/Tie)": "",  # To be filled by human rater
            "score_A (1-5)": "",        # To be filled by human rater
            "score_B (1-5)": "",        # To be filled by human rater
            "reviewer_notes": "",       # To be filled by human rater
        })

    # Write blank blind CSV for human reviewer
    fieldnames = [
        "sample_id",
        "language_pair",
        "source_script",
        "variant_A",
        "variant_B",
        "preferred (A/B/Tie)",
        "score_A (1-5)",
        "score_B (1-5)",
        "reviewer_notes",
    ]
    with open(sheet_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Write key mapping
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(key_mapping, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print(" BLIND RATING SHEET GENERATED")
    print("=" * 65)
    print(f"Rating Sheet (Fill this) : {sheet_path}")
    print(f"Answer Key (Do not edit) : {key_path}")
    print("\nInstructions for Human Reviewer:")
    print("1. Open the CSV file in Excel or any CSV editor.")
    print("2. For each row, read Variant A and Variant B.")
    print("3. Fill 'preferred (A/B/Tie)' with 'A', 'B', or 'Tie'.")
    print("4. Fill 'score_A (1-5)' and 'score_B (1-5)' with integer scores.")
    print("5. Run: python eval/score_ratings.py --sheet <sheet_path> --key <key_path>")
    print("=" * 65)


if __name__ == "__main__":
    samples_dir = Path("data/samples")
    default_samples = list(samples_dir.glob("*.txt"))
    if not default_samples:
        print("No sample files found in data/samples/.", file=sys.stderr)
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser(description="Create a blind side-by-side evaluation rating sheet.")
    parser.add_argument("--pair", default="hindi_to_hinglish", help="Language pair key")
    args = parser.parse_args()

    create_rating_sheet(sample_files=default_samples, pair_key=args.pair)
