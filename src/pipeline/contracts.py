from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TransactionRecord:
    transaction_id: str
    sender_id: str
    recipient_id: str
    transaction_type: str
    amount: float
    location: str
    payment_method: str
    timestamp: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureContext:
    transaction_id: str
    base_risk_score: float
    signals: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetRunResult:
    dataset_name: str
    dataset_path: Path
    total_transactions: int
    fraud_ids: list[str]
    output_file: Path
    session_id: str | None = None

