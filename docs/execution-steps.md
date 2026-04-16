# Execution Steps

## 1) Environment Setup (Windows PowerShell)
```powershell
Set-Location "D:\Projects\Hackathon\Reply-AI-Hackathon"
python -m pip install --upgrade pip
pip install -r .\req.txt
```

## 2) Configure Environment Variables
Create/update `.env` in project root with:
- `OPENROUTER_API_KEY=...`
- `LANGFUSE_PUBLIC_KEY=...`
- `LANGFUSE_SECRET_KEY=...`
- `LANGFUSE_HOST=https://challenges.reply.com/langfuse`
- `TEAM_NAME=your-team-name`

## 3) Smoke Test Langfuse Tracing
```powershell
python .\langfuse\main.py
```
Expected result: console output with a generated session ID and 3 traced LLM calls.

## 4) Build/Run the Actual Pipeline
- Implement missing modules in `src/pipeline/` (`data_loader`, `features`, `agents`, `model`, `pipeline`, `cli`).
- Run dataset inference and generate output as ASCII `.txt` with one transaction ID per line.

## 5) Submission Checklist
- Training: upload output `.txt` only.
- Evaluation: upload output `.txt` + full source code `.zip` + Langfuse Session ID.
- Double-check first evaluation submission before sending (only one attempt per dataset).
