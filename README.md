## Predictive Maintenance + Lot Genealogy

This project uses `uv` and a single professional `marimo` app to explore:

- `MetroPT-3` for maintenance risk, health scoring, and maintenance timing
- `EV Battery QC` for batch genealogy, defect hotspots, and supplier / line risk

The notebook is designed for an executive and operations audience, with a clean
dashboard-style layout rather than a research-style scratchpad.

## Quick Start

On macOS, `lightgbm` needs the OpenMP runtime. Install it once with:

```bash
brew install libomp
```

Then start the project:

```bash
uv sync
uv run python scripts/get_data.py
uv run marimo edit notebooks/maintenance_genealogy_app.py
```

To run the app in read-only mode:

```bash
uv run marimo run notebooks/maintenance_genealogy_app.py
```

## Data Layout

Place raw files in these folders:

- `data/raw/metropt3/`
- `data/raw/ev_battery_qc/`

The app will automatically detect the first supported `.csv`, `.parquet`, or
`.feather` file in each folder. If a folder is empty, the app falls back to a
demo dataset so you can still review the experience immediately.

See [data/README.md](data/README.md) for details.

## Download Data

MetroPT-3 is fetched from the official UCI source. EV Battery QC is fetched
from Kaggle using an API token supplied via the environment.

```bash
export KAGGLE_API_TOKEN=...
uv run python scripts/get_data.py
```

Useful flags:

```bash
uv run python scripts/get_data.py --metro-only
uv run python scripts/get_data.py --ev-only
uv run python scripts/get_data.py --force
```

`kaggle.json` is optional. The downloader reads `KAGGLE_API_TOKEN` directly and
never writes credentials to disk.

## Inspiration Sources

The app includes a source panel referencing the Kaggle work that inspired the
structure and storytelling:

- [MetroPT-3 | Data Import & EDA Starter](https://www.kaggle.com/code/joebeachcapital/metropt-3-data-import-eda-starter)
- [Notebook Predictive Maintenance and XAI](https://www.kaggle.com/code/chinmayadatt/notebook-predictive-maintenance-and-xai/notebook)
- [EV Battery QC code gallery](https://www.kaggle.com/datasets/kanchana1990/ev-battery-qc-synthetic-defect-dataset/code)
