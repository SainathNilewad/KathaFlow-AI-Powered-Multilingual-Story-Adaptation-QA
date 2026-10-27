"""CLI runner for Kuku Localization: Runs pipeline and baseline, prints QA flags, and saves outputs.
Usage:
    python run_cli.py --file data/samples/sample_script.txt --pair hindi_to_hinglish
"""
import sys
import argparse
from pathlib import Path

# UTF-8 encoding safeguard for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import get_pair_config, list_available_pairs
from src.localizer import localize_script
from src.baseline import run_baseline
from src.llm_client import LLMClient
from src.logger import create_session_id, log_run


def main():
    parser = argparse.ArgumentParser(description="Run Kuku Script Localization and Baseline comparison.")
    parser.add_argument("--file", "-f", default="data/samples/sample_script.txt", help="Path to plain text script file")
    parser.add_argument("--pair", "-p", default="hindi_to_hinglish", help="Language pair key")
    parser.add_argument("--output-dir", "-o", default="output", help="Directory to save outputs")
    args = parser.parse_args()

    script_path = Path(args.file)
    if not script_path.exists():
        print(f"File not found: {script_path}", file=sys.stderr)
        sys.exit(1)

    with open(script_path, "r", encoding="utf-8") as f:
        source_text = f.read()

    pair_cfg = get_pair_config(args.pair)
    client = LLMClient()
    session_id = create_session_id()

    print("=" * 60)
    print(f"KUKU LOCALIZATION RUNNER | {pair_cfg.display_name}")
    print(f"File: {script_path.name} | LLM: {client.provider.upper()} (Mock Mode: {client.is_mock()})")
    print("=" * 60)

    # 1. Pipeline
    print("[1/2] Running Structured Pipeline (Chunking -> Glossary -> QA)...")
    pipe_out = localize_script(source_text, pair_cfg, llm_client=client)

    # 2. Baseline
    print("[2/2] Running Plain Baseline Prompt...")
    base_out = run_baseline(source_text, pair_cfg, llm_client=client)

    # Log run locally
    stats = pipe_out.stats
    log_run(
        session_id=session_id,
        language_pair=pair_cfg.display_name,
        source_script=source_text,
        chunks_count=stats.get("chunks_count", 0),
        flags_count=stats.get("flags_count", 0),
        critical_flags=stats.get("critical_flags", 0),
        warning_flags=stats.get("warning_flags", 0),
        info_flags=0,
        is_mock=client.is_mock(),
        elapsed_seconds=stats.get("elapsed_seconds", 0.0),
    )

    # Save outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = script_path.stem

    pipe_file = out_dir / f"{stem}_pipeline_{args.pair}.txt"
    base_file = out_dir / f"{stem}_baseline_{args.pair}.txt"

    with open(pipe_file, "w", encoding="utf-8") as f:
        f.write(pipe_out.full_adapted_text)
    with open(base_file, "w", encoding="utf-8") as f:
        f.write(base_out.adapted_text)

    print("\nRESULTS:")
    print(f"  Pipeline Output : {pipe_file} ({stats.get('adapted_word_count')} words, {stats.get('elapsed_seconds')}s)")
    print(f"  Baseline Output : {base_file} ({base_out.stats.get('adapted_word_count')} words)")
    print(f"  QA Flags Raised : {len(pipe_out.flags)} (Critical: {stats.get('critical_flags')}, Warnings: {stats.get('warning_flags')})")

    for idx, flg in enumerate(pipe_out.flags[:5], 1):
        print(f"    [{idx}] [{flg.severity.upper()}] {flg.type}: {flg.short_explanation}")
    if len(pipe_out.flags) > 5:
        print(f"    ... and {len(pipe_out.flags) - 5} more flags.")
    print("=" * 60)
    print(f"[SUCCESS] Adaptation complete! Outputs saved in {out_dir}/")


if __name__ == "__main__":
    main()
