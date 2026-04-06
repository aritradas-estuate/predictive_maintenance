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

If you are on macOS and `lightgbm` fails to import with a missing
`libomp.dylib`, install the OpenMP runtime first:

```bash
brew install libomp
```

Then run:

```bash
uv sync
uv run python scripts/get_data.py
uv run marimo edit notebooks/maintenance_genealogy_app.py
uv run marimo run notebooks/maintenance_genealogy_app.py
```

## Automated Download

The project includes a `uv`-run downloader that places real data in the correct
folders automatically.

### MetroPT-3

Fetched from the official UCI archive into `data/raw/metropt3/`.

### EV Battery QC

Fetched from Kaggle into `data/raw/ev_battery_qc/`.

Use environment-based authentication:

```bash
export KAGGLE_API_TOKEN=...
uv run python scripts/get_data.py
```

Optional commands:

```bash
uv run python scripts/get_data.py --metro-only
uv run python scripts/get_data.py --ev-only
uv run python scripts/get_data.py --force
```

The downloader uses `KAGGLE_API_TOKEN` when present. A local `kaggle.json` file
is not required.
