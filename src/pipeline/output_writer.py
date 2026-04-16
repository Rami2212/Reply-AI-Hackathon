from __future__ import annotations

from pathlib import Path


def sanitize_fraud_ids(fraud_ids: list[str], total_transactions: int) -> list[str]:
    unique_ids = list(dict.fromkeys([x.strip() for x in fraud_ids if x and x.strip()]))
    if total_transactions <= 0:
        return []
    if len(unique_ids) >= total_transactions:
        return unique_ids[: max(total_transactions - 1, 1)]
    return unique_ids


def write_ascii_output(
    path: Path,
    fraud_ids: list[str],
    total_transactions: int,
    fallback_transaction_id: str | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    valid_ids = sanitize_fraud_ids(fraud_ids, total_transactions)
    if not valid_ids and fallback_transaction_id:
        valid_ids = [fallback_transaction_id]
    if not valid_ids:
        raise ValueError("Unable to write valid output: no fraud IDs and no fallback transaction ID")

    content = "\n".join(valid_ids) + "\n"
    content.encode("ascii", errors="strict")
    path.write_text(content, encoding="ascii", newline="\n")
    return path
