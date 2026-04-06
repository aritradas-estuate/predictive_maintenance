# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "lightgbm==4.6.0",
#     "marimo>=0.22.4",
#     "networkx==3.6.1",
#     "numpy==2.4.4",
#     "pandas==3.0.2",
#     "plotly==6.6.0",
#     "scikit-learn==1.8.0",
# ]
# ///
import marimo

__generated_with = "0.22.4"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import networkx as nx
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from lightgbm import LGBMClassifier
    from plotly.subplots import make_subplots
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    ROOT = Path(__file__).resolve().parents[1]
    RAW_DIR = ROOT / "data" / "raw"
    METRO_DIR = RAW_DIR / "metropt3"
    EV_DIR = RAW_DIR / "ev_battery_qc"
    NOTEBOOK_DATE = "April 6, 2026"
    return (
        ColumnTransformer,
        EV_DIR,
        LGBMClassifier,
        METRO_DIR,
        NOTEBOOK_DATE,
        OneHotEncoder,
        Pipeline,
        ROOT,
        SimpleImputer,
        average_precision_score,
        go,
        make_subplots,
        mo,
        np,
        nx,
        pd,
        px,
        roc_auc_score,
        train_test_split,
    )


@app.cell
def _(
    ColumnTransformer,
    EV_DIR,
    LGBMClassifier,
    METRO_DIR,
    OneHotEncoder,
    Pipeline,
    SimpleImputer,
    average_precision_score,
    go,
    make_subplots,
    np,
    nx,
    pd,
    px,
    roc_auc_score,
    train_test_split,
):
    def _normalize_name(name: str) -> str:
        return (
            name.strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )


    def find_data_file(directory):
        if not directory.exists():
            return None
        candidates = []
        for suffix in ("*.parquet", "*.feather", "*.csv"):
            candidates.extend(sorted(directory.glob(suffix)))
        return candidates[0] if candidates else None


    def load_table(path, target_rows=120_000):
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
            if len(df) > target_rows:
                step = max(1, len(df) // target_rows)
                df = df.iloc[::step].copy()
            return df
        if path.suffix == ".feather":
            df = pd.read_feather(path)
            if len(df) > target_rows:
                step = max(1, len(df) // target_rows)
                df = df.iloc[::step].copy()
            return df

        file_size_mb = path.stat().st_size / 1_000_000
        if file_size_mb < 30:
            return pd.read_csv(path)

        chunks = []
        for chunk in pd.read_csv(path, chunksize=50_000):
            step = max(1, len(chunk) // 350)
            chunks.append(chunk.iloc[::step].copy())
        return pd.concat(chunks, ignore_index=True)


    def first_match(columns, candidates):
        normalized = {_normalize_name(col): col for col in columns}
        for candidate in candidates:
            if candidate in normalized:
                return normalized[candidate]
        for col in columns:
            norm = _normalize_name(col)
            if any(candidate in norm for candidate in candidates):
                return col
        return None


    def generate_demo_metro_data():
        rng = np.random.default_rng(7)
        timestamps = pd.date_range("2026-01-01", periods=24 * 120, freq="h")
        assets = ["Compressor-A", "Compressor-B", "Compressor-C"]
        frames = []
        for index, asset in enumerate(assets):
            base_pressure = 96 - index * 2
            base_temp = 64 + index * 3
            base_current = 32 + index * 1.5
            df = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "asset_id": asset,
                    "air_pressure_bar": base_pressure
                    + 1.8 * np.sin(np.arange(len(timestamps)) / 48)
                    + rng.normal(0, 0.7, len(timestamps)),
                    "oil_temperature_c": base_temp
                    + 2.2 * np.sin(np.arange(len(timestamps)) / 36)
                    + rng.normal(0, 0.9, len(timestamps)),
                    "motor_current_a": base_current
                    + 1.1 * np.sin(np.arange(len(timestamps)) / 24)
                    + rng.normal(0, 0.4, len(timestamps)),
                    "vibration_mm_s": 3.6
                    + index * 0.25
                    + 0.25 * np.sin(np.arange(len(timestamps)) / 10)
                    + rng.normal(0, 0.12, len(timestamps)),
                    "power_kw": 108
                    + index * 4
                    + 3.4 * np.sin(np.arange(len(timestamps)) / 18)
                    + rng.normal(0, 0.9, len(timestamps)),
                    "oil_level_pct": 88
                    - 0.004 * np.arange(len(timestamps))
                    + rng.normal(0, 0.18, len(timestamps)),
                }
            )

            for start_day in (32, 71, 101):
                start = start_day * 24 + index * 8
                end = start + 36
                ramp = np.linspace(0, 1, end - start)
                df.loc[start:end - 1, "vibration_mm_s"] += 1.8 * ramp
                df.loc[start:end - 1, "motor_current_a"] += 2.4 * ramp
                df.loc[start:end - 1, "oil_temperature_c"] += 3.6 * ramp
                df.loc[start:end - 1, "air_pressure_bar"] -= 3.0 * ramp
                df.loc[start:end - 1, "oil_level_pct"] -= 2.3 * ramp
            frames.append(df)

        return pd.concat(frames, ignore_index=True)


    def prepare_metro_dataset():
        source_path = find_data_file(METRO_DIR)
        if source_path is None:
            df = generate_demo_metro_data()
            mode = "demo"
            source = "Built-in demo telemetry"
        else:
            df = load_table(source_path)
            mode = "file"
            source = source_path.name

        df = df.copy()
        df.columns = [col.strip() for col in df.columns]

        timestamp_col = first_match(
            df.columns,
            ["timestamp", "datetime", "date_time", "time", "recorded_at", "date"],
        )
        asset_col = first_match(
            df.columns,
            ["asset_id", "machine_id", "unit", "compressor", "equipment_id"],
        )

        if timestamp_col is None:
            timestamp_col = "timestamp"
            df[timestamp_col] = pd.date_range("2026-01-01", periods=len(df), freq="h")
        else:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
            df = df.loc[df[timestamp_col].notna()].copy()

        if asset_col is None:
            asset_col = "asset_id"
            df[asset_col] = "Primary Compressor"

        numeric_cols = [
            col
            for col in df.select_dtypes(include="number").columns
            if col not in {asset_col}
        ]
        if df.empty or len(numeric_cols) < 3:
            df = generate_demo_metro_data()
            timestamp_col = "timestamp"
            asset_col = "asset_id"
            numeric_cols = [
                "air_pressure_bar",
                "oil_temperature_c",
                "motor_current_a",
                "vibration_mm_s",
                "power_kw",
                "oil_level_pct",
            ]
            mode = "demo"
            source = "Built-in demo telemetry"

        variability = (
            df[numeric_cols].std(numeric_only=True).sort_values(ascending=False).index
        )
        sensor_cols = list(variability[:6])
        df = df.sort_values([asset_col, timestamp_col]).reset_index(drop=True)
        assets = sorted(df[asset_col].astype(str).unique().tolist())

        return {
            "df": df,
            "mode": mode,
            "source": source,
            "timestamp_col": timestamp_col,
            "asset_col": asset_col,
            "sensor_cols": sensor_cols,
            "assets": assets,
        }


    def robust_risk_series(df, sensor_cols):
        working = df.copy()
        signal_scores = []
        window = max(12, min(72, len(working) // 12 or 12))
        for col in sensor_cols:
            series = working[col].astype(float).interpolate(limit_direction="both")
            rolling_center = series.rolling(window, min_periods=4).median()
            rolling_std = (
                series.rolling(window, min_periods=4).std().replace(0, np.nan)
            )
            trend_score = ((series - rolling_center).abs() / rolling_std).clip(0, 6)
            delta = series.diff().abs()
            delta_center = delta.rolling(window, min_periods=4).median()
            delta_std = delta.rolling(window, min_periods=4).std().replace(0, np.nan)
            delta_score = ((delta - delta_center).abs() / delta_std).clip(0, 6)
            signal_scores.append((0.75 * trend_score + 0.25 * delta_score).fillna(0))

        stacked = pd.concat(signal_scores, axis=1)
        raw_risk = stacked.mean(axis=1)
        scale = max(raw_risk.quantile(0.95), 1)
        return (raw_risk / scale * 100).clip(0, 100)


    def analyze_metro(metro_state, selected_asset, horizon_hours):
        df = metro_state["df"]
        asset_col = metro_state["asset_col"]
        time_col = metro_state["timestamp_col"]
        sensor_cols = metro_state["sensor_cols"]

        asset_df = (
            df.loc[df[asset_col].astype(str) == selected_asset]
            .sort_values(time_col)
            .reset_index(drop=True)
        )
        risk = robust_risk_series(asset_df, sensor_cols)
        asset_df = asset_df.assign(risk_score=risk, health_score=100 - risk)
        latest = asset_df.iloc[-1]
        recent_window = asset_df.tail(min(len(asset_df), 24 * 7))
        mean_recent_risk = float(recent_window["risk_score"].mean())
        availability_proxy = max(55.0, 100.0 - mean_recent_risk * 0.55)
        runway_hours = int(max(8, horizon_hours * (1.35 - latest["risk_score"] / 100)))

        if latest["risk_score"] >= 78:
            recommendation = "Intervene now"
            maintenance_window = f"0-{max(8, horizon_hours // 4)}h"
        elif latest["risk_score"] >= 58:
            recommendation = "Plan next shift"
            maintenance_window = f"{max(4, horizon_hours // 6)}-{max(18, horizon_hours // 2)}h"
        else:
            recommendation = "Monitor and defer"
            maintenance_window = f"{max(12, horizon_hours // 2)}-{max(36, horizon_hours)}h"

        latest_driver_scores = {}
        for col in sensor_cols:
            series = asset_df[col].astype(float).interpolate(limit_direction="both")
            baseline = series.rolling(24, min_periods=4).median().fillna(series.median())
            deviation = abs(series.iloc[-1] - baseline.iloc[-1])
            latest_driver_scores[col] = float(deviation)

        driver_df = (
            pd.DataFrame(
                {
                    "signal": list(latest_driver_scores.keys()),
                    "magnitude": list(latest_driver_scores.values()),
                }
            )
            .sort_values("magnitude", ascending=False)
            .head(5)
        )

        view_cols = sensor_cols[:3]
        trend_fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.62, 0.38],
        )
        palette = ["#315c72", "#8a9ca8", "#cf8f3d"]
        for color, col in zip(palette, view_cols):
            trend_fig.add_trace(
                go.Scatter(
                    x=asset_df[time_col],
                    y=asset_df[col],
                    name=col,
                    line={"width": 2, "color": color},
                ),
                row=1,
                col=1,
            )

        trend_fig.add_trace(
            go.Scatter(
                x=asset_df[time_col],
                y=asset_df["risk_score"],
                name="risk_score",
                line={"width": 3, "color": "#b45309"},
                fill="tozeroy",
                fillcolor="rgba(180, 83, 9, 0.12)",
            ),
            row=2,
            col=1,
        )
        trend_fig.add_hrect(
            y0=70,
            y1=100,
            fillcolor="rgba(185, 28, 28, 0.12)",
            line_width=0,
            row=2,
            col=1,
        )
        trend_fig.update_layout(
            height=540,
            margin={"l": 20, "r": 20, "t": 30, "b": 10},
            paper_bgcolor="#f5f7f9",
            plot_bgcolor="#ffffff",
            legend={"orientation": "h", "y": 1.08},
            title="Asset telemetry and maintenance urgency",
        )
        trend_fig.update_yaxes(title_text="Telemetry", row=1, col=1)
        trend_fig.update_yaxes(title_text="Risk score", row=2, col=1, range=[0, 100])

        driver_fig = px.bar(
            driver_df,
            x="magnitude",
            y="signal",
            orientation="h",
            color="magnitude",
            color_continuous_scale=["#dbe5ea", "#315c72", "#b45309"],
            title="Current maintenance drivers",
        )
        driver_fig.update_layout(
            height=300,
            margin={"l": 20, "r": 20, "t": 40, "b": 10},
            paper_bgcolor="#f5f7f9",
            plot_bgcolor="#ffffff",
            coloraxis_showscale=False,
        )
        driver_fig.update_yaxes(categoryorder="total ascending")

        action_table = pd.DataFrame(
            [
                {
                    "Decision": recommendation,
                    "Window": maintenance_window,
                    "Healthy runway (h)": runway_hours,
                    "Operational availability": round(availability_proxy, 1),
                }
            ]
        )

        return {
            "asset_df": asset_df,
            "trend_fig": trend_fig,
            "driver_fig": driver_fig,
            "action_table": action_table,
            "risk_score": float(latest["risk_score"]),
            "health_score": float(latest["health_score"]),
            "availability_proxy": float(availability_proxy),
            "runway_hours": runway_hours,
            "recommendation": recommendation,
            "maintenance_window": maintenance_window,
            "driver_df": driver_df,
        }


    def generate_demo_ev_data():
        rng = np.random.default_rng(21)
        suppliers = ["Supplier-A", "Supplier-B", "Supplier-C"]
        lines = ["Line-1", "Line-2", "Line-3"]
        shifts = ["Day", "Swing", "Night"]
        batches = [f"BATCH-{idx:03d}" for idx in range(1, 61)]
        rows = []

        for batch in batches:
            supplier = suppliers[rng.integers(0, len(suppliers))]
            line = lines[rng.integers(0, len(lines))]
            shift = shifts[rng.integers(0, len(shifts))]
            temp_shift = {"Day": 0.3, "Swing": 1.2, "Night": -0.2}[shift]
            line_bias = {"Line-1": 0.0, "Line-2": 0.35, "Line-3": 0.75}[line]
            supplier_bias = {
                "Supplier-A": -0.1,
                "Supplier-B": 0.2,
                "Supplier-C": 0.55,
            }[supplier]

            for cell_idx in range(80):
                ambient = rng.normal(24.8 + temp_shift, 1.8)
                overhang = rng.normal(1.05 + line_bias * 0.1, 0.08)
                electrolyte = rng.normal(4.25 - supplier_bias * 0.12, 0.12)
                resistance = rng.normal(38 + supplier_bias * 3 + line_bias * 1.5, 2.4)
                capacity = rng.normal(2995 - supplier_bias * 45 - line_bias * 20, 35)
                retention = rng.normal(93.5 - supplier_bias * 1.5 - line_bias * 1.2, 1.4)

                risk = (
                    max(0, ambient - 26) * 0.18
                    + max(0, overhang - 1.08) * 4.5
                    + max(0, 4.15 - electrolyte) * 5.4
                    + max(0, resistance - 39) * 0.24
                    + max(0, 2975 - capacity) * 0.02
                    + max(0, 92.5 - retention) * 0.45
                )
                risk += rng.normal(0, 0.35)

                if risk > 3.6:
                    grade = "Scrap"
                    defect = "Sealing drift"
                    comment = "Electrolyte low and impedance unstable."
                elif risk > 1.9:
                    grade = "Grade B"
                    defect = "Alignment variance"
                    comment = "Borderline alignment trend observed."
                else:
                    grade = "Grade A"
                    defect = "None"
                    comment = "Within expected process band."

                rows.append(
                    {
                        "Cell_ID": f"{batch}-CELL-{cell_idx:03d}",
                        "Batch_ID": batch,
                        "Supplier": supplier,
                        "Production_Line": line,
                        "Shift": shift,
                        "Ambient_Temp_C": round(ambient, 3),
                        "Anode_Overhang_mm": round(overhang, 4),
                        "Electrolyte_Volume_ml": round(electrolyte, 4),
                        "Internal_Resistance_mOhm": round(resistance, 3),
                        "Capacity_mAh": round(capacity, 2),
                        "Retention_50Cycle_Pct": round(retention, 3),
                        "Inspector_Comment": comment,
                        "Defect_Type": defect,
                        "QC_Grade": grade,
                    }
                )
        return pd.DataFrame(rows)


    def prepare_ev_dataset():
        source_path = find_data_file(EV_DIR)
        if source_path is None:
            df = generate_demo_ev_data()
            mode = "demo"
            source = "Built-in genealogy demo"
        else:
            df = load_table(source_path, target_rows=40_000)
            mode = "file"
            source = source_path.name

        df = df.copy()
        df.columns = [col.strip() for col in df.columns]
        colmap = {_normalize_name(col): col for col in df.columns}
        required = {
            "batch": colmap.get("batch_id"),
            "supplier": colmap.get("supplier"),
            "line": colmap.get("production_line") or colmap.get("line"),
            "shift": colmap.get("shift"),
            "grade": colmap.get("qc_grade") or colmap.get("grade"),
            "defect": colmap.get("defect_type"),
        }

        if any(value is None for value in required.values()):
            df = generate_demo_ev_data()
            mode = "demo"
            source = "Built-in genealogy demo"
            required = {
                "batch": "Batch_ID",
                "supplier": "Supplier",
                "line": "Production_Line",
                "shift": "Shift",
                "grade": "QC_Grade",
                "defect": "Defect_Type",
            }

        return {
            "df": df,
            "mode": mode,
            "source": source,
            "batch_col": required["batch"],
            "supplier_col": required["supplier"],
            "line_col": required["line"],
            "shift_col": required["shift"],
            "grade_col": required["grade"],
            "defect_col": required["defect"],
        }


    def _ev_risk_model(ev_state):
        df = ev_state["df"].copy()
        grade_col = ev_state["grade_col"]
        feature_candidates = [
            "Ambient_Temp_C",
            "Anode_Overhang_mm",
            "Electrolyte_Volume_ml",
            "Internal_Resistance_mOhm",
            "Capacity_mAh",
            "Retention_50Cycle_Pct",
        ]
        numeric_cols = [col for col in feature_candidates if col in df.columns]
        categorical_cols = [
            ev_state["supplier_col"],
            ev_state["line_col"],
            ev_state["shift_col"],
        ]

        if not numeric_cols:
            df["predicted_risk"] = 0.5
            return (
                df,
                {"average_precision": np.nan, "roc_auc": np.nan},
                pd.DataFrame(columns=["feature", "importance"]),
            )

        target = (df[grade_col].astype(str) != "Grade A").astype(int)
        if target.nunique() < 2:
            df["predicted_risk"] = target.astype(float)
            return (
                df,
                {"average_precision": np.nan, "roc_auc": np.nan},
                pd.DataFrame(columns=["feature", "importance"]),
            )
        X = df[numeric_cols + categorical_cols].copy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, target, test_size=0.25, random_state=42, stratify=target
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", SimpleImputer(strategy="median"), numeric_cols),
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ),
                        ]
                    ),
                    categorical_cols,
                ),
            ]
        )

        model = Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=120,
                        learning_rate=0.06,
                        num_leaves=31,
                        random_state=42,
                        verbose=-1,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)

        test_probs = model.predict_proba(X_test)[:, 1]
        all_probs = model.predict_proba(X)[:, 1]
        df["predicted_risk"] = all_probs

        metrics = {
            "average_precision": float(average_precision_score(y_test, test_probs)),
            "roc_auc": float(roc_auc_score(y_test, test_probs)),
        }

        prep = model.named_steps["prep"]
        feature_names = prep.get_feature_names_out()
        importance_df = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": model.named_steps["model"].feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .head(10)
        )
        importance_df["feature"] = importance_df["feature"].str.replace(
            "numeric__", "", regex=False
        ).str.replace("categorical__encoder__", "", regex=False)

        return df, metrics, importance_df


    def lineage_figure(batch_rows, supplier_col, line_col, shift_col, batch_col, grade_col):
        if batch_rows.empty:
            return go.Figure()

        graph = nx.DiGraph()
        for _, row in batch_rows.iterrows():
            graph.add_edge(row[supplier_col], row[line_col])
            graph.add_edge(row[line_col], row[shift_col])
            graph.add_edge(row[shift_col], row[batch_col])
            graph.add_edge(row[batch_col], row[grade_col])

        node_layers = {
            "supplier": sorted(batch_rows[supplier_col].astype(str).unique().tolist()),
            "line": sorted(batch_rows[line_col].astype(str).unique().tolist()),
            "shift": sorted(batch_rows[shift_col].astype(str).unique().tolist()),
            "batch": sorted(batch_rows[batch_col].astype(str).unique().tolist()),
            "grade": sorted(batch_rows[grade_col].astype(str).unique().tolist()),
        }
        x_positions = {
            "supplier": 0.05,
            "line": 0.28,
            "shift": 0.5,
            "batch": 0.72,
            "grade": 0.94,
        }

        coords = {}
        node_text = {}
        layer_palette = {
            "supplier": "#315c72",
            "line": "#8aa1af",
            "shift": "#cf8f3d",
            "batch": "#6b7280",
            "grade": "#b45309",
        }
        for layer_name, nodes in node_layers.items():
            for idx, node in enumerate(nodes):
                y = 1 - (idx + 1) / (len(nodes) + 1)
                coords[node] = (x_positions[layer_name], y)
                node_text[node] = layer_name.title()

        edge_x = []
        edge_y = []
        for source, target in graph.edges():
            x0, y0 = coords[source]
            x1, y1 = coords[target]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line={"color": "#cbd5db", "width": 2},
                hoverinfo="none",
                showlegend=False,
            )
        )

        for layer_name, nodes in node_layers.items():
            fig.add_trace(
                go.Scatter(
                    x=[coords[node][0] for node in nodes],
                    y=[coords[node][1] for node in nodes],
                    mode="markers+text",
                    text=nodes,
                    textposition="top center",
                    marker={
                        "size": 18,
                        "color": layer_palette[layer_name],
                        "line": {"color": "#ffffff", "width": 1.5},
                    },
                    name=layer_name.title(),
                    hovertext=[f"{layer_name.title()}: {node}" for node in nodes],
                    hoverinfo="text",
                )
            )

        fig.update_layout(
            height=380,
            margin={"l": 20, "r": 20, "t": 40, "b": 20},
            paper_bgcolor="#f5f7f9",
            plot_bgcolor="#ffffff",
            title="Lot genealogy view",
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return fig


    def analyze_ev(ev_state, selected_supplier, selected_line, selected_batch):
        modeled_df, metrics, importance_df = _ev_risk_model(ev_state)
        supplier_col = ev_state["supplier_col"]
        line_col = ev_state["line_col"]
        shift_col = ev_state["shift_col"]
        batch_col = ev_state["batch_col"]
        grade_col = ev_state["grade_col"]
        defect_col = ev_state["defect_col"]

        filtered = modeled_df.copy()
        if selected_supplier != "All":
            filtered = filtered.loc[filtered[supplier_col].astype(str) == selected_supplier]
        if selected_line != "All":
            filtered = filtered.loc[filtered[line_col].astype(str) == selected_line]
        if selected_batch != "All":
            filtered = filtered.loc[filtered[batch_col].astype(str) == selected_batch]

        qc_mix = (
            filtered.groupby(grade_col).size().reset_index(name="count").sort_values(
                "count", ascending=False
            )
        )
        supplier_hotspots = (
            modeled_df.groupby(supplier_col)
            .agg(
                affected_share=(grade_col, lambda values: (values != "Grade A").mean()),
                avg_risk=("predicted_risk", "mean"),
            )
            .reset_index()
            .sort_values("avg_risk", ascending=False)
        )
        batch_summary = (
            modeled_df.groupby(batch_col)
            .agg(
                supplier=(supplier_col, "first"),
                line=(line_col, "first"),
                shift=(shift_col, "first"),
                avg_risk=("predicted_risk", "mean"),
                scrap_share=(grade_col, lambda values: (values == "Scrap").mean()),
            )
            .reset_index()
            .sort_values(["avg_risk", "scrap_share"], ascending=False)
        )

        focus_batch = selected_batch
        if focus_batch == "All":
            focus_batch = str(batch_summary.iloc[0][batch_col])
        focus_rows = modeled_df.loc[modeled_df[batch_col].astype(str) == focus_batch].copy()

        qc_fig = px.pie(
            qc_mix,
            names=grade_col,
            values="count",
            color=grade_col,
            color_discrete_map={
                "Grade A": "#315c72",
                "Grade B": "#cf8f3d",
                "Scrap": "#b91c1c",
            },
            hole=0.55,
            title="Quality mix for current filters",
        )
        qc_fig.update_layout(
            height=320,
            margin={"l": 20, "r": 20, "t": 40, "b": 20},
            paper_bgcolor="#f5f7f9",
        )

        hotspot_fig = px.bar(
            supplier_hotspots,
            x="avg_risk",
            y=supplier_col,
            color="affected_share",
            orientation="h",
            color_continuous_scale=["#dbe5ea", "#cf8f3d", "#b91c1c"],
            title="Supplier genealogy hotspots",
        )
        hotspot_fig.update_layout(
            height=320,
            margin={"l": 20, "r": 20, "t": 40, "b": 20},
            paper_bgcolor="#f5f7f9",
            plot_bgcolor="#ffffff",
            coloraxis_colorbar_title="Affected share",
        )

        importance_fig = px.bar(
            importance_df.sort_values("importance", ascending=True),
            x="importance",
            y="feature",
            orientation="h",
            title="Batch quality risk drivers",
            color="importance",
            color_continuous_scale=["#dbe5ea", "#315c72", "#b45309"],
        )
        importance_fig.update_layout(
            height=340,
            margin={"l": 20, "r": 20, "t": 40, "b": 20},
            paper_bgcolor="#f5f7f9",
            plot_bgcolor="#ffffff",
            coloraxis_showscale=False,
        )

        lineage_fig = lineage_figure(
            focus_rows[[supplier_col, line_col, shift_col, batch_col, grade_col]].drop_duplicates(),
            supplier_col,
            line_col,
            shift_col,
            batch_col,
            grade_col,
        )

        top_batch = batch_summary.iloc[0]
        defect_view = (
            focus_rows.groupby(defect_col)
            .size()
            .reset_index(name="cells")
            .sort_values("cells", ascending=False)
            .head(5)
        )
        defect_table = defect_view.rename(columns={defect_col: "Defect"})

        return {
            "modeled_df": modeled_df,
            "metrics": metrics,
            "qc_fig": qc_fig,
            "hotspot_fig": hotspot_fig,
            "importance_fig": importance_fig,
            "lineage_fig": lineage_fig,
            "batch_summary": batch_summary,
            "top_batch": top_batch,
            "focus_batch": focus_batch,
            "defect_table": defect_table,
        }


    metro_state = prepare_metro_dataset()
    ev_state = prepare_ev_dataset()
    return analyze_ev, analyze_metro, ev_state, metro_state


@app.cell(hide_code=True)
def _(NOTEBOOK_DATE, mo):
    mo.Html(
        """
        <style>
        :root {
          --pm-bg: #f5f7f9;
          --pm-panel: #ffffff;
          --pm-ink: #13212b;
          --pm-muted: #5a6b76;
          --pm-steel: #315c72;
          --pm-slate: #8aa1af;
          --pm-amber: #cf8f3d;
          --pm-alert: #b45309;
          --pm-border: #dbe3e8;
        }
        .pm-hero {
          padding: 1.4rem 1.6rem;
          border-radius: 22px;
          background:
            radial-gradient(circle at top right, rgba(207, 143, 61, 0.22), transparent 30%),
            linear-gradient(135deg, #13212b 0%, #315c72 55%, #8aa1af 100%);
          color: white;
          border: 1px solid rgba(255,255,255,0.16);
          box-shadow: 0 20px 44px rgba(19, 33, 43, 0.18);
        }
        .pm-hero p {
          margin: 0.35rem 0 0;
          color: rgba(255,255,255,0.86);
        }
        .pm-caption {
          color: var(--pm-muted);
          font-size: 0.95rem;
        }
        .pm-panel {
          background: var(--pm-panel);
          border: 1px solid var(--pm-border);
          border-radius: 18px;
          padding: 1rem 1.1rem;
          box-shadow: 0 14px 28px rgba(49, 92, 114, 0.06);
        }
        .pm-linklist a {
          color: var(--pm-steel);
        }
        </style>
        """
    )
    mo.Html(
        f"""
        <section class="pm-hero">
          <div style="font-size:0.86rem; letter-spacing:0.08em; text-transform:uppercase;">
            Predictive Maintenance Intelligence
          </div>
          <h1 style="margin:0.35rem 0 0; font-size:2rem;">
            Maintenance + genealogy command center
          </h1>
          <p>
            A professional marimo app for maintenance timing, asset health, and lot
            traceability. Refreshed with Kaggle notebook inspiration reviewed on
            {NOTEBOOK_DATE}.
          </p>
        </section>
        """
    )
    return


@app.cell(hide_code=True)
def _(ev_state, metro_state, mo):
    metro_note = (
        "Demo mode: no MetroPT-3 file detected in data/raw/metropt3."
        if metro_state["mode"] == "demo"
        else f"Using MetroPT-3 source: {metro_state['source']}."
    )
    ev_note = (
        "Demo mode: no EV Battery QC file detected in data/raw/ev_battery_qc."
        if ev_state["mode"] == "demo"
        else f"Using EV Battery QC source: {ev_state['source']}."
    )
    mo.callout(
        mo.md(
            f"""
            **Data readiness**

            - {metro_note}
            - {ev_note}
            - The app still runs end-to-end in demo mode so the executive story and
              visual design stay reviewable before data onboarding.
            """
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(ev_state, metro_state, mo):
    asset_picker = mo.ui.dropdown(
        metro_state["assets"],
        value=metro_state["assets"][0],
        label="selected_asset",
        full_width=True,
    )
    horizon_slider = mo.ui.slider(
        start=12,
        stop=168,
        step=12,
        value=48,
        label="maintenance_horizon_hours",
        show_value=True,
        full_width=True,
    )
    suppliers = ["All"] + sorted(
        ev_state["df"][ev_state["supplier_col"]].astype(str).unique().tolist()
    )
    supplier_picker = mo.ui.dropdown(
        suppliers,
        value="All",
        label="selected_supplier",
        full_width=True,
    )
    lines = ["All"] + sorted(ev_state["df"][ev_state["line_col"]].astype(str).unique().tolist())
    line_picker = mo.ui.dropdown(
        lines,
        value="All",
        label="selected_line",
        full_width=True,
    )
    controls = mo.hstack(
        [asset_picker, horizon_slider, supplier_picker, line_picker],
        widths="equal",
        gap=1.0,
        wrap=True,
        align="stretch",
    )
    mo.vstack(
        [
            mo.md("### Decision controls"),
            controls,
        ],
        gap=0.6,
    )
    return asset_picker, horizon_slider, line_picker, supplier_picker


@app.cell(hide_code=True)
def _(ev_state, line_picker, mo, supplier_picker):
    filtered = ev_state["df"]
    if supplier_picker.value != "All":
        filtered = filtered.loc[
            filtered[ev_state["supplier_col"]].astype(str) == supplier_picker.value
        ]
    if line_picker.value != "All":
        filtered = filtered.loc[
            filtered[ev_state["line_col"]].astype(str) == line_picker.value
        ]

    batches = ["All"] + sorted(filtered[ev_state["batch_col"]].astype(str).unique().tolist())
    batch_picker = mo.ui.dropdown(
        batches,
        value="All",
        label="selected_batch",
        full_width=True,
    )
    mo.hstack(
        [
            mo.md("### Genealogy focus"),
            batch_picker,
        ],
        widths=[1, 2],
        gap=1.0,
        align="center",
    )
    return (batch_picker,)


@app.cell
def _(
    analyze_ev,
    analyze_metro,
    asset_picker,
    batch_picker,
    ev_state,
    horizon_slider,
    line_picker,
    metro_state,
    supplier_picker,
):
    metro_analysis = analyze_metro(
        metro_state,
        selected_asset=asset_picker.value,
        horizon_hours=horizon_slider.value,
    )
    ev_analysis = analyze_ev(
        ev_state,
        selected_supplier=supplier_picker.value,
        selected_line=line_picker.value,
        selected_batch=batch_picker.value,
    )
    return ev_analysis, metro_analysis


@app.cell(hide_code=True)
def _(ev_analysis, metro_analysis, mo):
    kpis = mo.hstack(
        [
            mo.stat(
                value=f"{metro_analysis['risk_score']:.0f}",
                label="maintenance urgency",
                caption=metro_analysis["recommendation"],
                bordered=True,
            ),
            mo.stat(
                value=f"{metro_analysis['health_score']:.0f}",
                label="asset health score",
                caption=f"window {metro_analysis['maintenance_window']}",
                bordered=True,
            ),
            mo.stat(
                value=f"{metro_analysis['availability_proxy']:.0f}%",
                label="usable availability",
                caption=f"runway {metro_analysis['runway_hours']}h",
                bordered=True,
            ),
            mo.stat(
                value=f"{ev_analysis['top_batch']['avg_risk']:.2f}",
                label="highest batch risk",
                caption=ev_analysis["focus_batch"],
                bordered=True,
            ),
        ],
        widths="equal",
        gap=1.0,
        wrap=True,
    )
    kpis
    return


@app.cell(hide_code=True)
def _(ROOT, mo):
    inspiration = mo.md(
        f"""
        ### Kaggle inspiration used in this app

        <div class="pm-linklist">

        - [MetroPT-3 | Data Import & EDA Starter](https://www.kaggle.com/code/joebeachcapital/metropt-3-data-import-eda-starter): inspired the time-series framing and clean telemetry-first story.
        - [Notebook Predictive Maintenance and XAI](https://www.kaggle.com/code/chinmayadatt/notebook-predictive-maintenance-and-xai/notebook): inspired the maintenance-driver and explainability lens.
        - [EV Battery QC code gallery](https://www.kaggle.com/datasets/kanchana1990/ev-battery-qc-synthetic-defect-dataset/code): used as the live reference point for batch-level storytelling and QC drilldowns.

        </div>

        Data placement details live in [`data/README.md`]({ROOT / "data" / "README.md"}).
        """
    )
    inspiration
    return


@app.cell(hide_code=True)
def _(ev_analysis, metro_analysis, mo):
    executive_view = mo.vstack(
        [
            mo.md(
                f"""
                ### Executive overview

                - **Maintenance posture:** {metro_analysis['recommendation']} for the selected asset, with an action window of **{metro_analysis['maintenance_window']}**.
                - **Lot genealogy posture:** batch **{ev_analysis['focus_batch']}** is the current focus lot, and supplier / line patterns indicate where defect containment should tighten first.
                - **Why this matters:** the app ties near-term machine stress to downstream quality outcomes so operations can coordinate maintenance and traceability instead of treating them as separate decisions.
                """
            ),
            mo.ui.table(metro_analysis["action_table"], selection=None, page_size=5),
        ],
        gap=0.8,
    )

    metro_view = mo.vstack(
        [
            metro_analysis["trend_fig"],
            metro_analysis["driver_fig"],
            mo.ui.table(metro_analysis["driver_df"], selection=None, page_size=5),
        ],
        gap=0.8,
    )

    genealogy_view = mo.vstack(
        [
            mo.hstack(
                [ev_analysis["qc_fig"], ev_analysis["hotspot_fig"]],
                widths="equal",
                gap=1.0,
                wrap=True,
                align="stretch",
            ),
            ev_analysis["lineage_fig"],
            mo.ui.table(ev_analysis["defect_table"], selection=None, page_size=5),
        ],
        gap=0.8,
    )

    model_view = mo.vstack(
        [
            mo.md(
                f"""
                ### Model summary

                - EV quality model average precision: **{ev_analysis['metrics']['average_precision']:.2f}**
                - EV quality model ROC AUC: **{ev_analysis['metrics']['roc_auc']:.2f}**
                - Metro maintenance logic uses a rolling anomaly profile to transform telemetry drift into an operational risk score and maintenance window recommendation.
                """
            ),
            ev_analysis["importance_fig"],
        ],
        gap=0.8,
    )

    recommendations = mo.md(
        f"""
        ### Operational recommendations

        1. Schedule the selected asset inside **{metro_analysis['maintenance_window']}** if current staffing permits; the latest maintenance urgency is **{metro_analysis['risk_score']:.0f}/100**.
        2. Review the process settings tied to batch **{ev_analysis['focus_batch']}**, then compare them against the highest-risk supplier genealogy path.
        3. Capture maintenance events against batch identifiers going forward so the next version can directly quantify how maintenance timing shifts downstream scrap and downgrade rates.
        """
    )

    app_tabs = mo.ui.tabs(
        {
            "Executive overview": executive_view,
            "MetroPT-3 maintenance": metro_view,
            "Lot genealogy": genealogy_view,
            "Risk drivers": model_view,
            "Recommendations": recommendations,
        },
        value="Executive overview",
    )
    app_tabs
    return


if __name__ == "__main__":
    app.run()
