# Reply-AI-Hackathon

## Current Focus
This repository currently keeps the decision and observability implementation in:
- `src/pipeline/agents.py`
- `src/pipeline/ramitha.py`
- `src/pipeline/config.py`
- `src/pipeline/langfuse_utils.py`

## Install Dependencies
```powershell
Set-Location "D:\Projects\Hackathon\Reply-AI-Hackathon"
python -m pip install --upgrade pip
pip install -r .\requirements.txt
```

## Notes
- Root environment file is loaded from `.env` by `src/pipeline/config.py`.
- Next implementation items are tracked in `docs/TODO.md`.
