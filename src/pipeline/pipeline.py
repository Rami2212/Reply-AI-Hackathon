from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from .agents import OrchestratorAgent
from .config import LangfuseConfig
from .contracts import DatasetRunResult


def discover_train_dataset_paths(data_root: Path) -> list[Path]:
    """Return dataset folders that directly contain transactions.csv."""
    paths = sorted({p.parent for p in data_root.glob("**/transactions.csv")}, key=lambda x: str(x).lower())
    return [p for p in paths if p.is_dir()]


def _dataset_slug(dataset_path: Path) -> str:
    return dataset_path.name.strip().replace(" ", "_").replace("-", "_").lower()


def _prepare_dataset_for_legacy_loader(dataset_path: Path, staging_root: Path) -> Path:
    """Adapt train datasets to the column format expected by locked ingestion/context agents."""
    staging_path = staging_root / _dataset_slug(dataset_path)
    staging_path.mkdir(parents=True, exist_ok=True)

    src_csv = dataset_path / "transactions.csv"
    dst_csv = staging_path / "transactions.csv"
    df = pd.read_csv(src_csv)

    rename_map = {
        "transaction_id": "Transaction ID",
        "sender_id": "Sender ID",
        "recipient_id": "Recipient ID",
        "transaction_type": "Transaction Type",
        "amount": "Amount",
        "location": "Location",
        "payment_method": "Payment Method",
        "sender_iban": "Sender IBAN",
        "recipient_iban": "Recipient IBAN",
        "balance_after": "Balance",
        "description": "Description",
        "timestamp": "Timestamp",
    }
    df = df.rename(columns=rename_map)
    df.to_csv(dst_csv, index=False)

    for name in ["users.json", "locations.json", "sms.json", "mails.json"]:
        src = dataset_path / name
        if src.exists():
            shutil.copy2(src, staging_path / name)

    _normalize_users_for_context_agent(df, staging_path / "users.json")
    return staging_path


def _normalize_users_for_context_agent(df: pd.DataFrame, users_path: Path) -> None:
    if not users_path.exists():
        return

    try:
        users = json.loads(users_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(users, list):
        return

    iban_to_sender: dict[str, str] = {}
    if "Sender IBAN" in df.columns and "Sender ID" in df.columns:
        pairs = df[["Sender IBAN", "Sender ID"]].dropna().drop_duplicates()
        for _, pair in pairs.iterrows():
            iban_to_sender[str(pair["Sender IBAN"]).strip()] = str(pair["Sender ID"]).strip()

    changed = False
    for record in users:
        if not isinstance(record, dict):
            continue
        if record.get("id"):
            continue

        iban = str(record.get("iban") or "").strip()
        sender_id = iban_to_sender.get(iban)
        if sender_id:
            record["id"] = sender_id
        elif iban:
            record["id"] = f"iban::{iban}"
        else:
            record["id"] = "unknown"
        changed = True

    if changed:
        users_path.write_text(json.dumps(users, ensure_ascii=True), encoding="utf-8")


def _read_output_ids(output_file: Path) -> list[str]:
    if not output_file.exists():
        return []
    return [line.strip() for line in output_file.read_text(encoding="ascii", errors="ignore").splitlines() if line.strip()]


def _ensure_langfuse_config_compat() -> None:
    # Keep locked config/langfuse_utils untouched; patch compatibility at runtime.
    if not hasattr(LangfuseConfig, "enabled"):
        setattr(
            LangfuseConfig,
            "enabled",
            property(lambda self: bool(getattr(self, "public_key", "") and getattr(self, "secret_key", ""))),
        )


def run_dataset(dataset_path: Path, output_dir: Path, allow_llm_review: bool = False) -> DatasetRunResult:
    dataset_slug = _dataset_slug(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    staging_path = _prepare_dataset_for_legacy_loader(dataset_path, output_dir / "_staging")
    output_file = output_dir / f"{dataset_slug}_fraud_ids.txt"

    _ensure_langfuse_config_compat()
    orchestrator = OrchestratorAgent(dataset_path=str(staging_path), output_path=str(output_file))
    if allow_llm_review is not None:
        orchestrator.decision_agent.allow_llm_review = bool(allow_llm_review)
    orchestrator.run()

    tx_count = len(pd.read_csv(staging_path / "transactions.csv"))
    fraud_ids = _read_output_ids(output_file)

    return DatasetRunResult(
        dataset_name=dataset_path.name,
        dataset_path=dataset_path,
        total_transactions=tx_count,
        fraud_ids=fraud_ids,
        output_file=output_file,
        session_id=getattr(orchestrator.obs, "session_id", None),
    )


def run_all_train_datasets(data_root: Path, output_dir: Path, allow_llm_review: bool = False) -> list[DatasetRunResult]:
    dataset_paths = discover_train_dataset_paths(data_root)
    if not dataset_paths:
        raise FileNotFoundError(f"No dataset folders with transactions.csv found under {data_root}")

    return [run_dataset(p, output_dir=output_dir, allow_llm_review=allow_llm_review) for p in dataset_paths]
