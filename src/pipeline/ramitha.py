from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .config import load_project_env
from .langfuse_utils import run_llm_call, setup_langfuse_integration


@dataclass
class DecisionInput:
    transaction_id: str
    sender_id: str
    recipient_id: str
    amount: float
    txn_type: str
    payment_method: str
    base_risk_score: float
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionResult:
    transaction_id: str
    predicted_fraud: int
    confidence: float
    reason: str
    used_llm: bool
    timestamp_utc: str


class DecisionAgent:
    """Agent 3: combines deterministic score with optional LLM tie-break on borderline cases."""

    def __init__(
            self,
            low_threshold: float = 0.35,
            high_threshold: float = 0.65,
            allow_llm_review: bool = True,
    ) -> None:
        if not 0 <= low_threshold < high_threshold <= 1:
            raise ValueError("Thresholds must satisfy 0 <= low < high <= 1")

        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.allow_llm_review = allow_llm_review
        self.model = None
        self.langfuse_client = None
        self.session_id = None

        # Ensure root .env is loaded even for deterministic-only mode.
        load_project_env()

        if self.allow_llm_review:
            try:
                self.model, self.langfuse_client, self.session_id = setup_langfuse_integration()
            except Exception:
                self.allow_llm_review = False

    def _build_prompt(self, decision_input: DecisionInput) -> str:
        context_preview = {
            "transaction_id": decision_input.transaction_id,
            "sender_id": decision_input.sender_id,
            "recipient_id": decision_input.recipient_id,
            "amount": decision_input.amount,
            "txn_type": decision_input.txn_type,
            "payment_method": decision_input.payment_method,
            "base_risk_score": round(decision_input.base_risk_score, 4),
            "signals": decision_input.context,
        }
        return (
            "You are a fraud decision reviewer for MirrorPay. "
            "Return only one character: 1 for FRAUD, 0 for LEGIT. "
            "Prioritize minimizing false positives while catching likely fraud. "
            f"Case: {context_preview}"
        )

    def _llm_borderline_decision(self, decision_input: DecisionInput) -> tuple[int, str]:
        if not self.allow_llm_review or self.model is None or self.session_id is None:
            return (1 if decision_input.base_risk_score >= 0.5 else 0, "borderline-no-llm")

        prompt = self._build_prompt(decision_input)
        content = (run_llm_call(self.session_id, self.model, prompt) or "").strip()
        if content.startswith("1"):
            return (1, "borderline-llm-fraud")
        if content.startswith("0"):
            return (0, "borderline-llm-legit")
        return (1 if decision_input.base_risk_score >= 0.5 else 0, "borderline-llm-unexpected")

    def decide(self, decision_input: DecisionInput) -> DecisionResult:
        risk = max(0.0, min(1.0, float(decision_input.base_risk_score)))
        used_llm = False

        if risk <= self.low_threshold:
            predicted = 0
            reason = "deterministic-low-risk"
        elif risk >= self.high_threshold:
            predicted = 1
            reason = "deterministic-high-risk"
        else:
            predicted, reason = self._llm_borderline_decision(decision_input)
            used_llm = reason.startswith("borderline-llm")

        confidence = abs(risk - 0.5) * 2
        if used_llm:
            confidence = max(confidence, 0.55)

        return DecisionResult(
            transaction_id=decision_input.transaction_id,
            predicted_fraud=predicted,
            confidence=round(confidence, 4),
            reason=reason,
            used_llm=used_llm,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )


class ObservabilityLayer:
    """Agent 5: lightweight observability wrapper for decision events and run lifecycle."""

    def __init__(self) -> None:
        self.model = None
        self.langfuse_client = None
        self.session_id: str | None = None
        self.trace = None

    def start(self, run_name: str = "decision-agent-run") -> str:
        self.model, self.langfuse_client, self.session_id = setup_langfuse_integration()
        if hasattr(self.langfuse_client, "trace"):
            self.trace = self.langfuse_client.trace(
                name=run_name,
                session_id=self.session_id,
                input={"run_name": run_name},
            )
        return self.session_id

    def log_decision(self, decision_input: DecisionInput, decision_result: DecisionResult) -> None:
        payload_in = {
            "transaction_id": decision_input.transaction_id,
            "sender_id": decision_input.sender_id,
            "recipient_id": decision_input.recipient_id,
            "amount": decision_input.amount,
            "txn_type": decision_input.txn_type,
            "payment_method": decision_input.payment_method,
            "base_risk_score": decision_input.base_risk_score,
            "context": decision_input.context,
        }
        payload_out = {
            "predicted_fraud": decision_result.predicted_fraud,
            "confidence": decision_result.confidence,
            "reason": decision_result.reason,
            "used_llm": decision_result.used_llm,
            "timestamp_utc": decision_result.timestamp_utc,
        }

        if self.trace is not None and hasattr(self.trace, "event"):
            self.trace.event(name="decision", input=payload_in, output=payload_out)
            return

        if self.langfuse_client is not None and hasattr(self.langfuse_client, "trace"):
            self.langfuse_client.trace(
                name="decision",
                session_id=self.session_id,
                input=payload_in,
                output=payload_out,
            )

    def flush(self) -> None:
        if self.langfuse_client is not None and hasattr(self.langfuse_client, "flush"):
            self.langfuse_client.flush()

