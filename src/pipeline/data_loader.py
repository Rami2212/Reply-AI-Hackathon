import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from .langfuse_utils import observe_step

logger = logging.getLogger(__name__)


class IngestionAgent:
    """
    Ingestion Agent: The first point of contact for the 'Eye'.
    Responsible for loading multi-modal data and ensuring it is ready for
    high-level LLM reasoning.
    """

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.raw_data: Dict[str, Any] = {}

    @observe_step(name="IngestionAgent.run")
    def run(self) -> Dict[str, Any]:
        """Executes the full ingestion pipeline."""
        logger.info(f"Starting ingestion from: {self.dataset_path}")

        # 1. Load the backbone: Transactions
        self.raw_data['transactions'] = self._load_transactions()

        # 2. Load auxiliary context (Behavioral/Semantic)
        self.raw_data['users'] = self._load_json_file("users.json")
        self.raw_data['locations'] = self._load_json_file("locations.json")
        self.raw_data['sms'] = self._load_json_file("sms.json")
        self.raw_data['mails'] = self._load_json_file("mails.json")

        # 3. Validation as per Scoring Rules [cite: 93-96]
        self._check_basic_validity()

        return self.raw_data

    @observe_step(name="IngestionAgent.load_transactions")
    def _load_transactions(self) -> pd.DataFrame:
        """Loads and prepares the transaction CSV [cite: 55-72]."""
        file_path = self.dataset_path / "transactions.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Mandatory Transactions.csv missing at {file_path}")

        df = pd.read_csv(file_path)

        # Ensure Timestamps are actual datetime objects for temporal analysis
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        logger.info(f"Loaded {len(df)} transactions.")
        return df

    @observe_step(name="IngestionAgent.load_json_context")
    def _load_json_file(self, filename: str) -> List[Dict[str, Any]]:
        """Loads auxiliary JSON data (Users, Locations, SMS, Mails) [cite: 76-86]."""
        file_path = self.dataset_path / filename
        if not file_path.exists():
            logger.warning(f"Optional context file {filename} not found.")
            return []

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Loaded {len(data)} records from {filename}.")
            return data

    def _check_basic_validity(self):
        """Internal guardrails to prevent 'Invalid Output' scenarios [cite: 93-96]."""
        tx_df = self.raw_data.get('transactions')
        if tx_df is None or tx_df.empty:
            raise ValueError("The dataset contains no transactions. System cannot proceed.")

        # Check for essential columns required for tracking [cite: 56, 63]
        required = {'Transaction ID', 'Sender ID', 'Amount'}
        if not required.issubset(tx_df.columns):
            raise ValueError(f"Transaction file missing mandatory fields: {required - set(tx_df.columns)}")
