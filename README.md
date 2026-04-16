# Reply-AI-Hackathon

## What Is Ready
- Pipeline runner: `src/pipeline/pipeline.py`
- CLI entrypoint: `src/pipeline/cli.py` and `python -m src.pipeline`
- Smoke harness: `tests/smoke_pipeline.py`

## Install Dependencies
```powershell
Set-Location "D:\Projects\Hackathon\Reply-AI-Hackathon"
python -m pip install --upgrade pip
pip install -r .\requirements.txt
```

## Run All 3 Train Datasets
```powershell
Set-Location "D:\Projects\Hackathon\Reply-AI-Hackathon"
py -3 -m src.pipeline --data-root data --output-dir outputs
```

## Run Smoke Harness
```powershell
Set-Location "D:\Projects\Hackathon\Reply-AI-Hackathon"
py -3 tests\smoke_pipeline.py
```

## Outputs
- One ASCII file per dataset in `outputs/` with one `transaction_id` per line.
- Staging copies for compatibility are written to `outputs/_staging/`.

## Notes
- I did not change `src/pipeline/agents.py`, `src/pipeline/config.py`, `src/pipeline/data_loader.py`, or `src/pipeline/langfuse_utils.py`.
- Root `.env` loading behavior remains controlled by existing config/agents code.
