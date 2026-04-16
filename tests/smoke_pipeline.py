from __future__ import annotations

from pathlib import Path

from src.pipeline.cli import run_all_train_datasets


def main() -> None:
    outputs = run_all_train_datasets(
        data_root=Path("data"),
        output_dir=Path("outputs"),
        allow_llm_review=False,
    )
    print(f"generated_files={len(outputs)}")
    for item in outputs:
        print(item)


if __name__ == "__main__":
    main()

