from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_all_train_datasets as run_all_train_datasets_with_results


def run_all_train_datasets(data_root: Path, output_dir: Path, allow_llm_review: bool) -> list[Path]:
    results = run_all_train_datasets_with_results(
        data_root=data_root,
        output_dir=output_dir,
        allow_llm_review=allow_llm_review,
    )

    generated_files: list[Path] = []
    for result in results:
        generated_files.append(result.output_file)
        print(f"dataset={result.dataset_name}")
        print(f"transactions={result.total_transactions}")
        print(f"predicted_fraud={len(result.fraud_ids)}")
        print(f"output={result.output_file}")
        print(f"session_id={result.session_id}")
        print("-")

    return generated_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pipeline for training datasets")
    parser.add_argument("--data-root", default="data", help="Root folder containing dataset subfolders")
    parser.add_argument("--output-dir", default="outputs", help="Output folder for generated txt files")
    parser.add_argument("--llm", action="store_true", help="Enable borderline LLM review")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_all_train_datasets(
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        allow_llm_review=args.llm,
    )


if __name__ == "__main__":
    main()

