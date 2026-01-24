# Experimental ML Workflow

This directory contains experimental multi-model training and prediction scripts.
These scripts are intended for research and experimentation only.

## Usage

You can still run these scripts via CLI:

```bash
python -m experimental.training.cli
python -m experimental.prediction.live
python -m experimental.backtest.cli
```

## Guardrails

- DO NOT import from this package in production backend or frontend code.
- This code is not guaranteed to follow the production schema or output contract.
- Production uses the `ml/production/` (or `ML/src/production/`) module.
