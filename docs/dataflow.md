# Dataflow

## End-to-End Flow
1. **Input selection**
   - Choose dataset folder (train or evaluation package) containing transactional and auxiliary files.
2. **Load & normalize**
   - Parse `transactions.csv` plus `users.json`, `locations.json`, `sms.json`, `mails.json` (and `audio/` when available).
3. **Entity context build**
   - Join data by user identifiers and timestamps to build behavioral timelines.
4. **Risk scoring**
   - Apply rule/model signals for anomalies (amount, time shift, location inconsistency, interaction patterns).
5. **Borderline review (optional LLM)**
   - Send compact case payloads to OpenRouter for tie-break decisions.
6. **Aggregation & thresholding**
   - Produce final binary decisions per transaction.
7. **Output writer**
   - Export ASCII text file with one suspected fraudulent `Transaction ID` per line.
8. **Trace flush**
   - Send run/session telemetry to Langfuse for challenge auditability.

## Data Contracts (Minimum)
- **Primary key:** `Transaction ID` from `transactions.csv`.
- **Join keys:** user identifiers in transactions and auxiliary datasets.
- **Output contract:** non-empty, not all transactions, newline-separated IDs.

## Operational Notes
- Training datasets allow iterative submissions.
- Evaluation datasets accept only first submission per dataset.
- Keep low false positives while adapting to evolving fraud patterns.

