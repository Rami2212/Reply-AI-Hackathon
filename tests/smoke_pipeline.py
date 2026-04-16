from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.cli import run_all_train_datasets


def main() -> None:
    outputs = run_all_train_datasets(
        data_root=Path("data"),
        output_dir=Path("outputs"),
        allow_llm_review=False,
    )
    assert len(outputs) == 3, f"Expected 3 output files, got {len(outputs)}"
    for item in outputs:
        assert item.exists(), f"Missing output file: {item}"
        print(item)


if __name__ == "__main__":
    main()
