# TODO

## 0) Challenge Constraints (Must Keep)
- [ ] Keep solution agent-based (not a single static deterministic script).
- [ ] Keep output as ASCII `.txt` with one suspected `transaction_id` per line.
- [ ] Ensure output validity guards: not empty, not all transactions.
- [ ] Track LLM interactions through Langfuse when LLM mode is enabled.
- [ ] Keep runtime cost and latency under control (credits are limited).

## 1) Core Pipeline Completion
- [x] Agent 1 ingestion for `transactions.csv` (`src/pipeline/data_loader.py`).
- [x] Agent 2 lightweight context/risk features (`src/pipeline/features.py`).
- [x] Agent 3 decision logic + optional LLM tie-break (`src/pipeline/ramitha.py`).
- [x] Agent 4 orchestration (`src/pipeline/pipeline.py`).
- [x] Agent 5 observability hooks (`src/pipeline/ramitha.py`).
- [x] Batch runner for all training datasets (`src/pipeline/cli.py`).

## 2) Improve Detection Quality
- [ ] Add sender/recipient historical behavior features (velocity, amount drift, recurrence).
- [ ] Add geo-temporal consistency checks using `locations.json`.
- [ ] Add communication-derived risk signals from `sms.json` and `mails.json`.
- [ ] Add calibration/tuning for thresholds to reduce false positives.
- [ ] Add fallback strategy for borderline cases when LLM is unavailable.

## 3) Multimodal Expansion
- [ ] Parse `users.json` and include profile priors in feature context.
- [ ] Add optional audio metadata/ASR pipeline for `audio/` folders where present.
- [ ] Define normalized schema for all modalities and version it.

## 4) Reliability and Testing
- [ ] Add deterministic unit tests for `DecisionAgent` threshold branches.
- [ ] Add unit tests for output writer validity constraints.
- [ ] Add integration test for `run_all_train_datasets` over sample fixtures.
- [ ] Add Langfuse integration test (mocked) and one live smoke test.
- [ ] Add structured logging per dataset run (counts, timing, session ID).

## 5) Submission Readiness
- [ ] Produce one output file per dataset for train and evaluation runs.
- [ ] Add reproducible run instructions (single command for all datasets).
- [ ] Validate `.env` key names and required variables before run.
- [ ] Prepare evaluation submission package (`.zip` source + output + session ID).
- [ ] Add final pre-submit checklist (first evaluation submission is final per dataset).

## 6) Nice-to-Have
- [ ] Add score-tracking notebook/script for threshold sweeps on train data.
- [ ] Add configurable strategy presets: `fast`, `balanced`, `high-recall`.
- [ ] Add simple dashboard artifact summarizing run stats per dataset.
