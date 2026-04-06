# Data Placement

The `marimo` app auto-detects the first supported dataset file in each folder:

- `data/raw/metropt3/`
- `data/raw/ev_battery_qc/`

Supported formats:

- `.csv`
- `.parquet`
- `.feather`

## Recommended Files

### MetroPT-3

Place the MetroPT-3 telemetry export in:

```text
data/raw/metropt3/
```

The app looks for a timestamp column and numeric sensor columns automatically.
If your file is very large, the app downsamples it for interactivity.

### EV Battery QC

Place the EV Battery QC export in:

```text
data/raw/ev_battery_qc/
```

The app expects columns equivalent to:

- `Batch_ID`
- `Supplier`
- `Production_Line`
- `Shift`
- `QC_Grade`
- `Defect_Type`

Numeric process columns are detected automatically and used for risk scoring.

## Demo Mode

If either dataset folder is empty, the app generates a polished demo dataset so
the notebook still launches and demonstrates the intended workflow.

## Commands

```bash
uv sync
uv run marimo edit notebooks/maintenance_genealogy_app.py
uv run marimo run notebooks/maintenance_genealogy_app.py
```
