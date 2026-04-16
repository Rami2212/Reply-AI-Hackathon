import logging
from typing import Dict, Any
from .data_loader import IngestionAgent as DataLoader
from .langfuse_utils import observe_step, bind_session_to_current_trace

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
