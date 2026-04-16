from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .agents import ContextAgent, DecisionAgent, DecisionInput, IngestionAgent, ObservabilityLayer
from .contracts import DatasetRunResult
from .output_writer import write_ascii_output


def discover_train_dataset_paths(data_root: Path) -> list[Path]:
    """Return dataset folders that directly contain transactions.csv."""
    paths = sorted({p.parent for p in data_root.glob("**/transactions.csv")}, key=lambda x: str(x).lower())
    return [p for p in paths if p.is_dir()]


def _dataset_slug(dataset_path: Path) -> str:
    return dataset_path.name.strip().replace(" ", "_").replace("-", "_").lower()


def _build_session_id(dataset_slug: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{dataset_slug}-{stamp}"


def _prepare_dataset_for_legacy_loader(dataset_path: Path, staging_root: Path) -> Path:
    """
    Adapt train datasets to the column format expected by existing locked agents/data_loader.
    """
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


def _pick_value(row: pd.Series, keys: list[str], default: Any = "") -> Any:
    for key in keys:
        if key in row.index and pd.notna(row[key]):
            return row[key]
    return default


def _score_row(row: pd.Series) -> float:
    amount = float(_pick_value(row, ["Amount", "amount"], 0.0) or 0.0)
    payment_method = str(_pick_value(row, ["Payment Method", "payment_method"], "")).lower()
    is_night_tx = int(_pick_value(row, ["is_night_tx"], 0) or 0)

    amount_signal = min(max(amount, 0.0) / 5000.0, 1.0)
    payment_signal = 0.2 if payment_method in {"mobile device", "smartwatch"} else 0.05
    night_signal = 0.2 if is_night_tx == 1 else 0.0

    return round(min(1.0, amount_signal * 0.6 + payment_signal + night_signal), 4)


def run_dataset(dataset_path: Path, output_dir: Path, allow_llm_review: bool = False) -> DatasetRunResult:
    dataset_slug = _dataset_slug(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    staging_path = _prepare_dataset_for_legacy_loader(dataset_path, output_dir / "_staging")
    session_id = _build_session_id(dataset_slug)

    observability = ObservabilityLayer()
    try:
        session_id = observability.start(run_name=f"dataset-{dataset_slug}")
    except Exception:
        pass

    ingestion = IngestionAgent(dataset_path=str(staging_path), session_id=session_id)
    ingestion_state = ingestion.process()

    context_agent = ContextAgent()
    enriched_state = context_agent.enrich(ingestion_state)
    tx_df: pd.DataFrame = enriched_state["enriched_transactions"]

    decision_agent = DecisionAgent(allow_llm_review=allow_llm_review)

    fraud_ids: list[str] = []
    for _, row in tx_df.iterrows():
        decision_input = DecisionInput(
            transaction_id=str(_pick_value(row, ["Transaction ID", "transaction_id"], "")),
            sender_id=str(_pick_value(row, ["Sender ID", "sender_id"], "")),
            recipient_id=str(_pick_value(row, ["Recipient ID", "recipient_id"], "")),
            amount=float(_pick_value(row, ["Amount", "amount"], 0.0) or 0.0),
            txn_type=str(_pick_value(row, ["Transaction Type", "transaction_type"], "")),
            payment_method=str(_pick_value(row, ["Payment Method", "payment_method"], "")),
            base_risk_score=_score_row(row),
            context={
                "is_night_tx": int(_pick_value(row, ["is_night_tx"], 0) or 0),
                "hour": int(_pick_value(row, ["hour"], 0) or 0),
            },
        )
        decision = decision_agent.decide(decision_input)
        observability.log_decision(decision_input, decision)
        if decision.predicted_fraud == 1:
            fraud_ids.append(decision_input.transaction_id)

    fallback_id: str | None = None
    if not tx_df.empty:
        first_row = tx_df.iloc[0]
        fallback_id = str(first_row.get("Transaction ID") or first_row.get("transaction_id") or "")

    output_file = output_dir / f"{dataset_slug}_fraud_ids.txt"
    output_file = write_ascii_output(
        output_file,
        fraud_ids,
        total_transactions=len(tx_df),
        fallback_transaction_id=fallback_id,
    )

    observability.flush()

    return DatasetRunResult(
        dataset_name=dataset_path.name,
        dataset_path=dataset_path,
        total_transactions=len(tx_df),
        fraud_ids=fraud_ids,
        output_file=output_file,
        session_id=session_id,
    )


def run_all_train_datasets(data_root: Path, output_dir: Path, allow_llm_review: bool = False) -> list[DatasetRunResult]:
    dataset_paths = discover_train_dataset_paths(data_root)
    if not dataset_paths:
        raise FileNotFoundError(f"No dataset folders with transactions.csv found under {data_root}")

    return [run_dataset(p, output_dir=output_dir, allow_llm_review=allow_llm_review) for p in dataset_paths]


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
            # Deterministic fallback when sender mapping is unavailable.
            record["id"] = f"iban::{iban}"
        else:
            record["id"] = "unknown"
        changed = True

    if changed:
        users_path.write_text(json.dumps(users, ensure_ascii=True), encoding="utf-8")
