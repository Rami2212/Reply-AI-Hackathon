# Project Architecture (Reply AI Hackathon)

## Goal
Build an agent-based fraud detection system that ingests multi-modal data and outputs suspected fraudulent `Transaction ID`s as ASCII text (one ID per line).

## Current Repository Structure
- `data/`: training datasets (`transactions.csv`, `users.json`, `locations.json`, `sms.json`, `mails.json`, optional `audio/`).
- `guidelines/`: official challenge problem statement and rules.
- `src/pipeline/`: target production pipeline package (mostly scaffolding right now).
- `langfuse/main.py`: working Langfuse + OpenRouter tracing example.
- `langfuse/requirements.txt`: minimal Python dependencies for tracing demo.

## Logical Components
1. Ingestion Agent
   - Loads and validates all input modalities from one dataset folder.
2. Feature/Context Agent
   - Builds per-user and per-transaction context (history, location, communication, behavior drift).
3. Decision Agent
   - Combines rules/ML signals and optional LLM review for borderline cases.
4. Orchestrator Agent
   - Runs agent flow, manages thresholds, and produces final fraud ID list.
5. Observability Layer
   - Langfuse tracing for all LLM calls and run metadata (session-bound).

## Implemented vs Planned
- Implemented:
  - `src/pipeline/openrouter_client.py`: `OpenRouterDecisionClient` for optional LLM override.
  - `src/pipeline/langfuse_utils.py`: `TraceClient`, trace lifecycle, event emission helpers.
  - `langfuse/main.py`: reference script for session generation and traced calls.
- Planned (currently empty files):
  - `src/pipeline/{agents,cli,config,data_loader,features,model,pipeline,prompts}.py`.

## Challenge Constraints to Keep in Architecture
- Agent-based approach is mandatory.
- Output must be ASCII `.txt`, one suspicious transaction ID per line.
- For evaluation datasets, include full source code zip + execution instructions.
- Track LLM interactions with Langfuse (required for validation and scoring transparency).

