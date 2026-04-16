import logging
import os
from typing import Dict, Any
from .data_loader import IngestionAgent as DataLoader
from .langfuse_utils import observe_step, bind_session_to_current_trace
from dataclasses import dataclass, field
from datetime import datetime, timezone
from .config import load_project_env, build_openrouter_config, build_langfuse_config
from .langfuse_utils import build_trace_client
from .openrouter_client import OpenRouterDecisionClient
import pandas as pd

logger = logging.getLogger(__name__)


class IngestionAgent:
    """
    The Ingestion Agent: Acts as 'The Eye's' intake specialist.
    It orchestrates the loading process and ensures data satisfies the
    Challenge Problem Statement requirements.
    """

    def __init__(self, dataset_path: str, session_id: str):
        self.loader = DataLoader(dataset_path)
        self.session_id = session_id
        self.context: Dict[str, Any] = {}

    @observe_step(name="IngestionAgent.process")
    def process(self) -> Dict[str, Any]:
        """
        Main entry point for the agent to ingest and validate the data package.
        """
        # 1. Attach the official challenge session ID to this trace
        bind_session_to_current_trace(self.session_id)

        logger.info(f"[IngestionAgent] Beginning intake for session: {self.session_id}")

        try:
            # 2. Trigger the underlying loader
            raw_payload = self.loader.run()

            # 3. Transform raw data into a structured 'State' for the Context Agent
            self.context = {
                "session_id": self.session_id,
                "transactions": raw_payload['transactions'],
                "metadata": {
                    "user_registry": raw_payload['users'],
                    "geo_logs": raw_payload['locations'],
                    "comms": {
                        "sms": raw_payload['sms'],
                        "email": raw_payload['mails']
                    }
                },
                "stats": {
                    "total_tx": len(raw_payload['transactions']),
                    "has_geo": raw_payload['locations'] is not None,
                    "has_comms": raw_payload['sms'] is not None
                }
            }

            # 4. Perform Challenge-Specific Validation [cite: 93-96]
            self._validate_challenge_constraints()

            logger.info(f"[IngestionAgent] Successfully prepared context with "
                        f"{self.context['stats']['total_tx']} transactions.")

            return self.context

        except Exception as e:
            logger.error(f"[IngestionAgent] Critical Failure: {str(e)}")
            raise

    def _validate_challenge_constraints(self):
        """
        Enforces rules from the Problem Statement Section 3 [cite: 93-96].
        """
        tx_count = self.context['stats']['total_tx']

        # Rule: No transactions reported results in invalid output [cite: 94]
        if tx_count == 0:
            raise ValueError("Challenge Violation: Input dataset contains 0 transactions.")

        # Log basic info for Langfuse observability [cite: 105]
        logger.info(f"Ingestion validated: {tx_count} records ready for analysis.")


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
        self.decision_client = None

        # Ensure root .env is loaded even for deterministic-only mode.
        load_project_env()

        if self.allow_llm_review:
            try:
                self.decision_client = OpenRouterDecisionClient(build_openrouter_config())
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
        if not self.allow_llm_review or self.decision_client is None:
            return (1 if decision_input.base_risk_score >= 0.5 else 0, "borderline-no-llm")

        payload = {
            "TransactionID": decision_input.transaction_id,
            "SenderID": decision_input.sender_id,
            "RecipientID": decision_input.recipient_id,
            "Amount": decision_input.amount,
            "TransactionType": decision_input.txn_type,
            "PaymentMethod": decision_input.payment_method,
            "BaseRiskScore": decision_input.base_risk_score,
            "Signals": decision_input.context,
        }
        value = self.decision_client.review_borderline_case(payload)
        if value == 1:
            return (1, "borderline-llm-fraud")
        if value == 0:
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
        self.trace_client = None

    def start(self, run_name: str = "decision-agent-run") -> str:
        load_project_env()
        team = (os.getenv("TEAM_NAME") or "team").replace(" ", "-")
        self.session_id = f"{team}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        self.trace_client = build_trace_client(build_langfuse_config())
        self.trace_client.start_run(self.session_id)
        self.trace_client.event(name="run_started", input_payload={"run_name": run_name})
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

        if self.trace_client is not None:
            self.trace_client.event(name="decision", input_payload=payload_in, output_payload=payload_out)
            return

        if self.langfuse_client is not None and hasattr(self.langfuse_client, "trace"):
            self.langfuse_client.trace(
                name="decision",
                session_id=self.session_id,
                input=payload_in,
                output=payload_out,
            )

    def flush(self) -> None:
        if self.trace_client is not None:
            self.trace_client.flush()
            return
        if self.langfuse_client is not None and hasattr(self.langfuse_client, "flush"):
            self.langfuse_client.flush()




class ContextAgent:
    """
    Context Agent: The behavioral brain.
    Builds timelines and identifies temporal/geographic shifts[cite: 24, 25].
    """
    @observe_step(name="ContextAgent.enrich")
    def enrich(self, context: Dict[str, Any]) -> Dict[str, Any]:
        tx_df = context['transactions']
        meta = context['metadata']
        
        logger.info("Context Agent: Enriching behavioral data...")

        # 1. Temporal Analysis: Spotting 'Late-night activity' 
        tx_df['hour'] = tx_df['Timestamp'].dt.hour
        tx_df['is_night_tx'] = tx_df['hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)

        # 2. User Context: Merging demographics for 'MirrorPay' citizens [cite: 17]
        user_df = pd.DataFrame(meta['user_registry'])
        if not user_df.empty:
            # Joins based on Sender ID [cite: 56, 81]
            tx_df = tx_df.merge(user_df, left_on='Sender ID', right_on='id', how='left')

        # 3. Geo-Spatial Preparation: Mapping transaction locations [cite: 64, 76]
        # This allows the Decision Agent to check for 'geographic shifts' [cite: 25]
        
        return {
            "session_id": context['session_id'],
            "enriched_transactions": tx_df,
            "raw_comms": meta['comms'] # Pass to Decision Agent for LLM review
        }

