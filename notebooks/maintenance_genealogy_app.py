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

__generated_with = "0.23.1"
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


    def _downsample_for_plot(x, y, max_points=1500):
        """Downsample arrays to max_points using LTTB-like min/max preservation."""
        import numpy as np

        n = len(y) if hasattr(y, "__len__") else len(x)
        if n <= max_points:
            return x, y
        step = max(1, n // max_points)
        # For each bucket, keep the point with max absolute value to preserve peaks
        indices = []
        for start in range(0, n, step):
            end = min(start + step, n)
            chunk = (
                y[start:end] if hasattr(y, "__getitem__") else list(y)[start:end]
            )
            try:
                local_idx = start + np.argmax(
                    np.abs(np.array(chunk, dtype=float) - np.nanmean(chunk))
                )
            except (ValueError, TypeError):
                local_idx = start
            indices.append(local_idx)
        indices = np.array(indices)
        if hasattr(x, "iloc"):
            return x.iloc[indices], y.iloc[indices] if hasattr(
                y, "iloc"
            ) else np.array(y)[indices]
        return np.array(x)[indices], np.array(y)[indices]


    def apply_panel_layout(
        fig,
        *,
        title,
        height,
        top_margin=68,
        bottom_margin=20,
        legend_y=1.02,
        show_legend=True,
    ):
        fig.update_layout(
            height=height,
            margin={"l": 20, "r": 20, "t": top_margin, "b": bottom_margin},
            paper_bgcolor="#f5f7f9",
            plot_bgcolor="#ffffff",
            title={"text": title, "x": 0.03, "xanchor": "left"},
        )
        if show_legend:
            fig.update_layout(
                legend={
                    "orientation": "h",
                    "y": legend_y,
                    "x": 0.0,
                    "yanchor": "bottom",
                }
            )
        else:
            fig.update_layout(showlegend=False)
        return fig


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


    _METRO_SENSOR_LABELS = {
        "TP2": "Temp Probe 2",
        "TP3": "Temp Probe 3",
        "H1": "Humidity Sensor",
        "DV_pressure": "Discharge Valve Pressure",
        "Reservoirs": "Reservoir Pressure",
        "Oil_temperature": "Oil Temperature (\u00b0C)",
        "Motor_current": "Motor Current (A)",
        "oil_temperature_c": "Oil Temperature (\u00b0C)",
        "motor_current_a": "Motor Current (A)",
        "air_pressure_bar": "Air Pressure (bar)",
        "vibration_mm_s": "Vibration (mm/s)",
        "power_kw": "Power (kW)",
        "oil_level_pct": "Oil Level (%)",
    }


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
                    "COMP": 1.0,
                }
            )

            for start_day in (32, 71, 101):
                start = start_day * 24 + index * 8
                end = start + 36
                ramp = np.linspace(0, 1, end - start)
                df.loc[start : end - 1, "vibration_mm_s"] += 1.8 * ramp
                df.loc[start : end - 1, "motor_current_a"] += 2.4 * ramp
                df.loc[start : end - 1, "oil_temperature_c"] += 3.6 * ramp
                df.loc[start : end - 1, "air_pressure_bar"] -= 3.0 * ramp
                df.loc[start : end - 1, "oil_level_pct"] -= 2.3 * ramp
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
            df[timestamp_col] = pd.date_range(
                "2026-01-01", periods=len(df), freq="h"
            )
        else:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
            df = df.loc[df[timestamp_col].notna()].copy()

        if asset_col is None:
            asset_col = "asset_id"
            df[asset_col] = "Primary Compressor"

        # Exclude metadata/index columns
        _skip = {asset_col}
        for col in df.columns:
            norm = _normalize_name(col)
            if any(tag in norm for tag in ("unnamed", "index", "row_id")):
                _skip.add(col)

        numeric_cols = [
            col
            for col in df.select_dtypes(include="number").columns
            if col not in _skip
        ]

        # Separate continuous vs binary sensors
        continuous_cols = [c for c in numeric_cols if df[c].nunique() > 2]
        binary_cols = [c for c in numeric_cols if df[c].nunique() <= 2]

        # Detect compressor state column
        comp_col = None
        for candidate in ("COMP", "comp", "Compressor_State"):
            if candidate in df.columns:
                comp_col = candidate
                break

        if df.empty or len(continuous_cols) < 3:
            df = generate_demo_metro_data()
            timestamp_col = "timestamp"
            asset_col = "asset_id"
            continuous_cols = [
                "air_pressure_bar",
                "oil_temperature_c",
                "motor_current_a",
                "vibration_mm_s",
                "power_kw",
                "oil_level_pct",
            ]
            binary_cols = ["COMP"]
            comp_col = "COMP"
            mode = "demo"
            source = "Built-in demo telemetry"

        variability = (
            df[continuous_cols]
            .std(numeric_only=True)
            .sort_values(ascending=False)
            .index
        )
        sensor_cols = list(variability[:6])

        sensor_labels = {}
        for col in sensor_cols:
            sensor_labels[col] = _METRO_SENSOR_LABELS.get(
                col, col.replace("_", " ").title()
            )

        df = df.sort_values([asset_col, timestamp_col]).reset_index(drop=True)
        assets = sorted(df[asset_col].astype(str).unique().tolist())

        return {
            "df": df,
            "mode": mode,
            "source": source,
            "timestamp_col": timestamp_col,
            "asset_col": asset_col,
            "sensor_cols": sensor_cols,
            "binary_cols": binary_cols,
            "comp_col": comp_col,
            "sensor_labels": sensor_labels,
            "assets": assets,
        }


    def robust_risk_series(df, sensor_cols, comp_col=None):
        working = df.copy()
        is_running = (
            working[comp_col].astype(float) >= 0.5
            if comp_col and comp_col in working.columns
            else pd.Series(True, index=working.index)
        )
        signal_scores = []
        window = max(12, min(72, len(working) // 12 or 12))
        for col in sensor_cols:
            series = working[col].astype(float).interpolate(limit_direction="both")
            # Compute baselines only on running periods
            run_series = series.where(is_running)
            rolling_center = (
                run_series.rolling(window, min_periods=4).median().ffill().bfill()
            )
            rolling_std = (
                run_series.rolling(window, min_periods=4)
                .std()
                .replace(0, np.nan)
                .ffill()
                .bfill()
            )
            trend_score = ((series - rolling_center).abs() / rolling_std).clip(
                0, 6
            )
            delta = series.diff().abs()
            delta_center = delta.rolling(window, min_periods=4).median()
            delta_std = (
                delta.rolling(window, min_periods=4).std().replace(0, np.nan)
            )
            delta_score = ((delta - delta_center).abs() / delta_std).clip(0, 6)
            combined = (0.75 * trend_score + 0.25 * delta_score).fillna(0)
            # Dampen score during idle periods
            combined = combined.where(is_running, combined * 0.15)
            signal_scores.append(combined)

        stacked = pd.concat(signal_scores, axis=1)
        raw_risk = stacked.mean(axis=1)
        scale = max(raw_risk.quantile(0.95), 1)
        return (raw_risk / scale * 100).clip(0, 100)


    def _compute_degradation(asset_df, sensor_cols, time_col, comp_col=None):
        n = len(asset_df)
        smooth_window = max(12, n // 50)
        is_running = (
            asset_df[comp_col].astype(float) >= 0.5
            if comp_col and comp_col in asset_df.columns
            else pd.Series(True, index=asset_df.index)
        )
        results = {}
        for col in sensor_cols:
            series = (
                asset_df[col].astype(float).interpolate(limit_direction="both")
            )
            smoothed = series.rolling(smooth_window, min_periods=4).mean().bfill()
            # Fit trend on running periods only
            run_mask = is_running.values
            t = np.arange(n)
            run_t = t[run_mask]
            run_vals = smoothed.values[run_mask]
            if len(run_t) < 10:
                run_t = t
                run_vals = smoothed.values
            slope, intercept = np.polyfit(run_t, run_vals, 1)
            residuals = run_vals - (slope * run_t + intercept)
            residual_std = float(np.std(residuals)) if len(residuals) > 1 else 1.0
            run_series = series[is_running]
            nominal_low = float(run_series.quantile(0.05))
            nominal_high = float(run_series.quantile(0.95))
            current_value = float(series.iloc[-1])
            results[col] = {
                "slope": float(slope),
                "intercept": float(intercept),
                "residual_std": residual_std,
                "nominal_low": nominal_low,
                "nominal_high": nominal_high,
                "nominal_center": (nominal_low + nominal_high) / 2,
                "nominal_half_range": (nominal_high - nominal_low) / 2,
                "current_value": current_value,
                "trend_line": slope * t + intercept,
                "smoothed": smoothed.values,
            }
        return results


    def _estimate_rul(asset_df, risk_series, degradation, sensor_cols, time_col):
        n = len(asset_df)
        # Data-driven risk threshold
        risk_threshold = max(70.0, float(risk_series.quantile(0.90)))

        # Fit linear trend on last 30% of risk series
        tail_start = max(0, int(n * 0.7))
        tail_risk = risk_series.values[tail_start:]
        tail_t = np.arange(len(tail_risk))
        if len(tail_t) >= 2:
            risk_slope, risk_intercept = np.polyfit(tail_t, tail_risk, 1)
        else:
            risk_slope, risk_intercept = 0.0, float(risk_series.iloc[-1])

        # Extrapolate: steps until risk hits threshold
        current_risk = float(risk_series.iloc[-1])
        if risk_slope > 0.001:
            steps_to_threshold = max(
                0, (risk_threshold - current_risk) / risk_slope
            )
        else:
            steps_to_threshold = float("inf")

        # Estimate hours per step from timestamp spacing
        if time_col in asset_df.columns and len(asset_df) >= 2:
            ts = pd.to_datetime(asset_df[time_col])
            total_hours = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 3600
            hours_per_step = total_hours / max(1, n - 1)
        else:
            hours_per_step = 1.0

        composite_rul_hours = (
            min(9999, steps_to_threshold * hours_per_step)
            if steps_to_threshold != float("inf")
            else 9999
        )

        # Per-sensor RUL: time to exit nominal band
        per_sensor_rul = {}
        for col in sensor_cols:
            deg = degradation[col]
            slope = deg["slope"]
            current = deg["current_value"]
            if abs(slope) < 1e-9:
                per_sensor_rul[col] = 9999
                continue
            if slope > 0:
                boundary = deg["nominal_high"]
                steps = (
                    max(0, (boundary - current) / slope)
                    if current < boundary
                    else 0
                )
            else:
                boundary = deg["nominal_low"]
                steps = (
                    max(0, (current - boundary) / abs(slope))
                    if current > boundary
                    else 0
                )
            per_sensor_rul[col] = min(9999, steps * hours_per_step)

        # Weakest link
        min_sensor_rul = min(per_sensor_rul.values()) if per_sensor_rul else 9999
        composite_rul_hours = min(composite_rul_hours, min_sensor_rul)

        weakest_sensor = (
            min(per_sensor_rul, key=per_sensor_rul.get) if per_sensor_rul else ""
        )

        return {
            "composite_rul_hours": int(composite_rul_hours),
            "per_sensor_rul": per_sensor_rul,
            "risk_threshold": risk_threshold,
            "risk_slope": float(risk_slope),
            "hours_per_step": hours_per_step,
            "weakest_sensor": weakest_sensor,
        }


    def _optimize_maintenance(
        rul_hours, horizon_hours, risk_score, risk_threshold
    ):
        n_candidates = 48
        candidates = np.linspace(0, horizon_hours, n_candidates)
        steepness = 6.0 / max(1, rul_hours)
        failure_probs = 1.0 / (1.0 + np.exp(-steepness * (candidates - rul_hours)))
        productive_values = candidates / max(1, horizon_hours)
        expected_benefit = productive_values - 2.0 * failure_probs
        best_idx = int(np.argmax(expected_benefit))
        optimal_hour = float(candidates[best_idx])
        failure_at_optimal = float(failure_probs[best_idx])

        # Data-driven urgency thresholds
        high_threshold = risk_threshold * 0.85
        mid_threshold = risk_threshold * 0.65
        if risk_score >= high_threshold:
            recommendation = "Intervene now"
            window_lo = 0
            window_hi = max(8, int(optimal_hour))
        elif risk_score >= mid_threshold:
            recommendation = "Plan next shift"
            window_lo = max(4, int(optimal_hour * 0.5))
            window_hi = max(18, int(optimal_hour * 1.2))
        else:
            recommendation = "Monitor and defer"
            window_lo = max(12, int(optimal_hour * 0.8))
            window_hi = max(36, int(optimal_hour * 1.5))
        maintenance_window = f"{window_lo}-{window_hi}h"

        return {
            "optimal_hour": int(optimal_hour),
            "failure_at_optimal": failure_at_optimal,
            "recommendation": recommendation,
            "maintenance_window": maintenance_window,
            "candidates": candidates,
            "failure_probs": failure_probs,
            "expected_benefit": expected_benefit,
        }


    def _compute_usability(degradation, sensor_cols):
        per_sensor = {}
        weights = []
        usabilities = []
        for col in sensor_cols:
            deg = degradation[col]
            half_range = deg["nominal_half_range"]
            if half_range < 1e-9:
                per_sensor[col] = 100.0
                continue
            dist = abs(deg["current_value"] - deg["nominal_center"])
            usability = max(0.0, (1.0 - dist / half_range)) * 100
            per_sensor[col] = round(usability, 1)
            usabilities.append(usability)
            weights.append(abs(deg["slope"]))

        total_weight = sum(weights)
        if total_weight > 0 and usabilities:
            composite = (
                sum(u * w for u, w in zip(usabilities, weights)) / total_weight
            )
        elif usabilities:
            composite = sum(usabilities) / len(usabilities)
        else:
            composite = 100.0

        weakest = min(per_sensor, key=per_sensor.get) if per_sensor else ""
        return {
            "composite_pct": round(composite, 1),
            "per_sensor": per_sensor,
            "weakest_sensor": weakest,
        }


    def _build_degradation_fig(
        asset_df, degradation, sensor_cols, time_col, sensor_labels
    ):
        n_sensors = min(len(sensor_cols), 4)
        cols = sensor_cols[:n_sensors]
        fig = make_subplots(
            rows=n_sensors,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=[sensor_labels.get(c, c) for c in cols],
        )
        palette = ["#315c72", "#cf8f3d", "#8a9ca8", "#6b7280"]
        t_vals = (
            asset_df[time_col]
            if time_col in asset_df.columns
            else list(range(len(asset_df)))
        )
        for i, col in enumerate(cols, 1):
            deg = degradation[col]
            ds_t, ds_smooth = _downsample_for_plot(t_vals, deg["smoothed"])
            fig.add_trace(
                go.Scatter(
                    x=ds_t,
                    y=ds_smooth,
                    name=sensor_labels.get(col, col),
                    line={"width": 2, "color": palette[i - 1]},
                    showlegend=(i == 1),
                    legendgroup=col,
                ),
                row=i,
                col=1,
            )
            ds_t2, ds_trend = _downsample_for_plot(t_vals, deg["trend_line"])
            fig.add_trace(
                go.Scatter(
                    x=ds_t2,
                    y=ds_trend,
                    name="Trend",
                    line={"width": 2, "color": palette[i - 1], "dash": "dash"},
                    showlegend=(i == 1),
                    legendgroup="trend",
                ),
                row=i,
                col=1,
            )
            # Nominal range band
            fig.add_hrect(
                y0=deg["nominal_low"],
                y1=deg["nominal_high"],
                fillcolor="rgba(49, 92, 114, 0.08)",
                line_width=0,
                row=i,
                col=1,
            )
            # Confidence band around trend
            upper = deg["trend_line"] + deg["residual_std"]
            lower = deg["trend_line"] - deg["residual_std"]
            ds_t3, ds_upper = _downsample_for_plot(t_vals, upper)
            _, ds_lower = _downsample_for_plot(t_vals, lower)
            fig.add_trace(
                go.Scatter(
                    x=list(ds_t3) + list(ds_t3)[::-1],
                    y=list(ds_upper) + list(ds_lower)[::-1],
                    fill="toself",
                    fillcolor=f"rgba(49, 92, 114, 0.06)",
                    line={"width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=i,
                col=1,
            )
        apply_panel_layout(
            fig,
            title="Sensor degradation trends with nominal bands",
            height=200 * n_sensors,
            top_margin=68,
            show_legend=False,
        )
        return fig


    def _build_schedule_fig(schedule):
        candidates = schedule["candidates"]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=candidates,
                y=schedule["failure_probs"] * 100,
                name="Failure probability",
                line={"width": 2, "color": "#b91c1c"},
                fill="tozeroy",
                fillcolor="rgba(185, 28, 28, 0.08)",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=candidates,
                y=schedule["expected_benefit"],
                name="Expected benefit",
                line={"width": 2, "color": "#315c72"},
            ),
            secondary_y=True,
        )
        opt = schedule["optimal_hour"]
        fig.add_vline(
            x=opt,
            line_dash="dash",
            line_color="#cf8f3d",
            line_width=2,
            annotation_text=f"Optimal: {opt}h",
            annotation_position="top right",
            annotation_yshift=-12,
        )
        apply_panel_layout(
            fig,
            title="Maintenance schedule optimizer",
            height=380,
            top_margin=76,
            legend_y=1.03,
        )
        fig.update_layout(xaxis_title="Hours from now")
        fig.update_yaxes(title_text="Failure probability (%)", secondary_y=False)
        fig.update_yaxes(title_text="Expected benefit", secondary_y=True)
        return fig


    def _build_usability_fig(usability, sensor_labels):
        per_sensor = usability["per_sensor"]
        labels = [sensor_labels.get(c, c) for c in per_sensor]
        values = list(per_sensor.values())
        colors = [
            "#315c72" if v >= 70 else "#cf8f3d" if v >= 40 else "#b91c1c"
            for v in values
        ]
        fig = go.Figure(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker_color=colors,
                text=[f"{v:.0f}%" for v in values],
                textposition="auto",
            )
        )
        fig.add_vline(
            x=usability["composite_pct"],
            line_dash="dash",
            line_color="#6b7280",
            annotation_text=f"Composite: {usability['composite_pct']:.0f}%",
        )
        apply_panel_layout(
            fig,
            title="Asset usability by sensor",
            height=max(200, 50 * len(per_sensor)),
            top_margin=68,
            show_legend=False,
        )
        fig.update_layout(xaxis={"range": [0, 105], "title": "Usability (%)"})
        return fig


    def _compute_warning_profile(current_risk, rul_info, schedule):
        risk_threshold = float(rul_info["risk_threshold"])
        warning_threshold = risk_threshold * 0.65
        critical_threshold = risk_threshold * 0.85
        slope = max(float(rul_info["risk_slope"]), 0.0)
        hours_per_step = max(float(rul_info["hours_per_step"]), 1e-6)

        def _hours_to(target):
            if current_risk >= target:
                return 0.0
            if slope <= 1e-6:
                return float(rul_info["composite_rul_hours"])
            return max(0.0, (target - current_risk) / slope * hours_per_step)

        warning_horizon = min(
            float(rul_info["composite_rul_hours"]), _hours_to(warning_threshold)
        )
        intervention_horizon = min(
            float(rul_info["composite_rul_hours"]), _hours_to(critical_threshold)
        )

        if (
            current_risk >= critical_threshold
            or schedule["recommendation"] == "Intervene now"
        ):
            state = "Intervention required"
        elif current_risk >= warning_threshold or warning_horizon <= 24:
            state = "Early warning"
        else:
            state = "Healthy runway"

        return {
            "warning_threshold": warning_threshold,
            "critical_threshold": critical_threshold,
            "warning_horizon_hours": int(round(warning_horizon)),
            "intervention_horizon_hours": int(round(intervention_horizon)),
            "state": state,
        }


    def _build_warning_timeline_fig(warning_profile):
        warning_h = max(0, warning_profile["warning_horizon_hours"])
        intervention_h = max(
            warning_h, warning_profile["intervention_horizon_hours"]
        )
        state = warning_profile["state"]
        xmax = max(intervention_h + 12, 24)

        fig = go.Figure()
        if state == "Intervention required":
            fig.add_trace(
                go.Bar(
                    y=["Runway"],
                    x=[max(1, intervention_h or 1)],
                    orientation="h",
                    marker={"color": "#b91c1c"},
                    name="Intervene",
                    text=["Intervene now"],
                    textposition="inside",
                )
            )
        else:
            healthy_h = max(0, min(warning_h, intervention_h))
            warning_window = max(0, intervention_h - healthy_h)
            if healthy_h > 0:
                fig.add_trace(
                    go.Bar(
                        y=["Runway"],
                        x=[healthy_h],
                        orientation="h",
                        marker={"color": "#315c72"},
                        name="Healthy",
                        text=[f"{healthy_h}h healthy"],
                        textposition="inside",
                    )
                )
            if warning_window > 0:
                fig.add_trace(
                    go.Bar(
                        y=["Runway"],
                        x=[warning_window],
                        base=[healthy_h],
                        orientation="h",
                        marker={"color": "#cf8f3d"},
                        name="Warning",
                        text=[f"{warning_window}h warning"],
                        textposition="inside",
                    )
                )
            fig.add_trace(
                go.Bar(
                    y=["Runway"],
                    x=[max(2, xmax - intervention_h)],
                    base=[intervention_h],
                    orientation="h",
                    marker={"color": "rgba(185, 28, 28, 0.18)"},
                    name="Intervention zone",
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

        apply_panel_layout(
            fig,
            title="Warning horizon",
            height=240,
            top_margin=78,
            bottom_margin=20,
            legend_y=1.03,
        )
        fig.update_layout(
            barmode="overlay",
            xaxis={"title": "Hours from now", "range": [0, xmax]},
            yaxis={"visible": False},
        )
        fig.add_annotation(
            x=min(xmax - 1, intervention_h),
            y=0,
            text=f"{state}: {warning_profile['intervention_horizon_hours']}h to intervention boundary",
            showarrow=False,
            yshift=14,
            xanchor="right",
            font={"size": 13, "color": "#13212b"},
        )
        return fig


    def _build_failure_progression_fig(precursor_df):
        if precursor_df.empty:
            return go.Figure()

        fig = px.bar(
            precursor_df.sort_values("Hours to limit", ascending=False),
            x="Hours to limit",
            y="Signal",
            orientation="h",
            color="Trend",
            color_discrete_map={
                "Rising": "#b45309",
                "Falling": "#315c72",
                "Stable": "#8aa1af",
            },
            text="Sequence",
            title="Failure precursor ladder",
        )
        apply_panel_layout(
            fig,
            title="Failure precursor ladder",
            height=340,
            top_margin=74,
            legend_y=1.03,
        )
        fig.update_traces(textposition="outside")
        return fig


    def _build_regime_fig(regime_summary):
        if regime_summary.empty:
            return go.Figure()

        fig = px.bar(
            regime_summary,
            x="Regime",
            y="Mean risk",
            color="Share of time (%)",
            color_continuous_scale=["#dbe5ea", "#8aa1af", "#315c72"],
            text="Share label",
            title="Operating regime lens",
        )
        apply_panel_layout(
            fig,
            title="Operating regime",
            height=320,
            top_margin=64,
            show_legend=False,
        )
        fig.update_layout(coloraxis_colorbar_title="Share of time")
        fig.update_traces(textposition="outside")
        return fig


    def _build_portfolio_fig(portfolio_table, x_col="RUL (h)"):
        if portfolio_table.empty:
            return go.Figure()

        plot_df = portfolio_table.copy()
        label_idx = (
            portfolio_table.sort_values(
                ["Risk score", "RUL (h)"], ascending=[False, True]
            )
            .head(3)
            .index
        )
        plot_df["label"] = np.where(
            plot_df.index.isin(label_idx), plot_df["Maintenance item"], ""
        )
        fig = px.scatter(
            plot_df,
            x=x_col,
            y="Risk score",
            size="Usability (%)",
            color="Priority",
            color_discrete_map={
                "Intervene now": "#b91c1c",
                "Plan next shift": "#cf8f3d",
                "Monitor and defer": "#315c72",
            },
            hover_name="Maintenance item",
            text="label",
            title="Maintenance priority board",
        )
        fig.update_traces(textposition="top center", textfont_size=11)
        apply_panel_layout(
            fig,
            title="Maintenance priority board",
            height=380,
            top_margin=78,
            legend_y=1.03,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="#b91c1c", opacity=0.6)
        fig.add_vline(
            x=max(12, float(portfolio_table[x_col].median())),
            line_dash="dash",
            line_color="#6b7280",
            opacity=0.5,
        )
        return fig


    def _build_economics_fig(economics_table):
        if economics_table.empty:
            return go.Figure()

        fig = go.Figure()
        palette = {
            "Planned downtime": "#315c72",
            "Failure exposure": "#b91c1c",
            "Avoidable disruption": "#cf8f3d",
        }
        for column in [
            "Planned downtime",
            "Failure exposure",
            "Avoidable disruption",
        ]:
            fig.add_trace(
                go.Bar(
                    x=economics_table["Scenario"],
                    y=economics_table[column],
                    name=column,
                    marker={"color": palette[column]},
                )
            )
        apply_panel_layout(
            fig,
            title="Maintenance economics",
            height=360,
            top_margin=78,
            legend_y=1.03,
        )
        fig.update_layout(barmode="stack", yaxis_title="Relative cost index")
        return fig


    def analyze_metro(metro_state, selected_asset, horizon_hours):
        df = metro_state["df"]
        asset_col = metro_state["asset_col"]
        time_col = metro_state["timestamp_col"]
        sensor_cols = metro_state["sensor_cols"]
        comp_col = metro_state.get("comp_col")
        sensor_labels = metro_state.get("sensor_labels", {})

        def _summarize_entity(entity_df, label, context):
            entity_df = entity_df.sort_values(time_col).reset_index(drop=True)
            risk = robust_risk_series(entity_df, sensor_cols, comp_col=comp_col)
            entity_df = entity_df.assign(risk_score=risk, health_score=100 - risk)
            latest = entity_df.iloc[-1]
            degradation = _compute_degradation(
                entity_df, sensor_cols, time_col, comp_col
            )
            rul_info = _estimate_rul(
                entity_df, risk, degradation, sensor_cols, time_col
            )
            schedule = _optimize_maintenance(
                rul_info["composite_rul_hours"],
                horizon_hours,
                float(latest["risk_score"]),
                rul_info["risk_threshold"],
            )
            usability = _compute_usability(degradation, sensor_cols)
            warning_profile = _compute_warning_profile(
                float(latest["risk_score"]), rul_info, schedule
            )

            latest_driver_scores = {}
            for col in sensor_cols:
                series = (
                    entity_df[col]
                    .astype(float)
                    .interpolate(limit_direction="both")
                )
                baseline = (
                    series.rolling(24, min_periods=4)
                    .median()
                    .fillna(series.median())
                )
                latest_driver_scores[col] = float(
                    abs(series.iloc[-1] - baseline.iloc[-1])
                )

            driver_df = (
                pd.DataFrame(
                    {
                        "signal": [
                            sensor_labels.get(col, col)
                            for col in latest_driver_scores
                        ],
                        "magnitude": list(latest_driver_scores.values()),
                        "source_col": list(latest_driver_scores.keys()),
                    }
                )
                .sort_values("magnitude", ascending=False)
                .head(6)
            )

            precursor_rows = []
            per_sensor_rul = rul_info["per_sensor_rul"]
            for idx, col in enumerate(
                sorted(per_sensor_rul, key=per_sensor_rul.get), start=1
            ):
                slope = degradation[col]["slope"]
                precursor_rows.append(
                    {
                        "Sequence": f"#{idx}",
                        "Signal": sensor_labels.get(col, col),
                        "Hours to limit": round(float(per_sensor_rul[col]), 1),
                        "Trend": "Rising"
                        if slope > 0
                        else "Falling"
                        if slope < 0
                        else "Stable",
                        "Slope": round(float(abs(slope)), 4),
                        "Usability (%)": round(
                            float(usability["per_sensor"].get(col, 100.0)), 1
                        ),
                    }
                )
            precursor_df = pd.DataFrame(precursor_rows)

            comp_series = (
                entity_df[comp_col].astype(float).fillna(0.0)
                if comp_col and comp_col in entity_df.columns
                else pd.Series(1.0, index=entity_df.index)
            )
            transition_mask = comp_series.diff().abs().fillna(0).gt(0)
            regime = np.where(
                transition_mask,
                "Transition",
                np.where(comp_series >= 0.5, "Loaded", "Idle"),
            )
            regime_df = entity_df.assign(regime=regime)
            regime_summary = (
                regime_df.groupby("regime")
                .agg(
                    **{
                        "Mean risk": ("risk_score", "mean"),
                        "Peak risk": ("risk_score", "max"),
                        "Hours": ("risk_score", "size"),
                    }
                )
                .reset_index()
                .rename(columns={"regime": "Regime"})
            )
            regime_summary["Share of time (%)"] = (
                regime_summary["Hours"] / max(1, len(regime_df)) * 100
            ).round(1)
            regime_summary["Share label"] = regime_summary[
                "Share of time (%)"
            ].map(lambda v: f"{v:.0f}%")
            regime_summary["Mean risk"] = regime_summary["Mean risk"].round(1)
            regime_summary["Peak risk"] = regime_summary["Peak risk"].round(1)

            incident_summary = {
                "what_changed": ", ".join(driver_df["signal"].head(2).tolist()),
                "where": f"{label} | weakest signal: {sensor_labels.get(rul_info['weakest_sensor'], rul_info['weakest_sensor'])}",
                "consequence": f"Current risk {float(latest['risk_score']):.0f}/100 with {rul_info['composite_rul_hours']}h remaining runway.",
                "recommended_action": f"{schedule['recommendation']} within {schedule['maintenance_window']}",
            }

            return {
                "entity_df": entity_df,
                "risk_score": float(latest["risk_score"]),
                "health_score": float(latest["health_score"]),
                "degradation": degradation,
                "rul_info": rul_info,
                "schedule": schedule,
                "usability": usability,
                "warning_profile": warning_profile,
                "driver_df": driver_df,
                "precursor_df": precursor_df,
                "regime_summary": regime_summary,
                "incident_summary": incident_summary,
                "context": context,
                "label": label,
            }

        asset_df = df.loc[df[asset_col].astype(str) == selected_asset].copy()
        summary = _summarize_entity(asset_df, selected_asset, "Selected asset")
        asset_df = summary["entity_df"]
        latest = asset_df.iloc[-1]
        degradation = summary["degradation"]
        rul_info = summary["rul_info"]
        schedule = summary["schedule"]
        usability = summary["usability"]
        driver_df = summary["driver_df"]
        precursor_df = summary["precursor_df"]
        regime_summary = summary["regime_summary"]
        warning_profile = summary["warning_profile"]
        incident_summary = summary["incident_summary"]

        portfolio_rows = []
        portfolio_note = ""
        assets = metro_state.get("assets", [])
        if len(assets) > 1:
            for asset_name in assets:
                entity_df = df.loc[df[asset_col].astype(str) == asset_name].copy()
                if len(entity_df) < 24:
                    continue
                entity_summary = _summarize_entity(
                    entity_df, str(asset_name), "Asset portfolio"
                )
                portfolio_rows.append(
                    {
                        "Maintenance item": str(asset_name),
                        "Context": entity_summary["context"],
                        "Risk score": round(entity_summary["risk_score"], 1),
                        "RUL (h)": int(
                            entity_summary["rul_info"]["composite_rul_hours"]
                        ),
                        "Warning horizon (h)": int(
                            entity_summary["warning_profile"][
                                "warning_horizon_hours"
                            ]
                        ),
                        "Usability (%)": entity_summary["usability"][
                            "composite_pct"
                        ],
                        "Priority": entity_summary["schedule"]["recommendation"],
                    }
                )
        else:
            split_indices = [
                idx_slice
                for idx_slice in np.array_split(asset_df.index.to_numpy(), 6)
                if len(idx_slice) >= 24
            ]
            slices = [
                asset_df.loc[idx_slice].copy() for idx_slice in split_indices
            ]
            portfolio_note = "Portfolio board is synthesized from maintenance backlog windows because the current raw file contains one physical asset."
            for idx, chunk in enumerate(slices, start=1):
                chunk = chunk.sort_values(time_col).reset_index(drop=True)
                start_ts = pd.to_datetime(chunk[time_col].iloc[0])
                end_ts = pd.to_datetime(chunk[time_col].iloc[-1])
                label = f"Backlog-{idx}"
                entity_summary = _summarize_entity(
                    chunk, label, f"{start_ts.date()} to {end_ts.date()}"
                )
                portfolio_rows.append(
                    {
                        "Maintenance item": label,
                        "Context": entity_summary["context"],
                        "Risk score": round(entity_summary["risk_score"], 1),
                        "RUL (h)": int(
                            entity_summary["rul_info"]["composite_rul_hours"]
                        ),
                        "Warning horizon (h)": int(
                            entity_summary["warning_profile"][
                                "warning_horizon_hours"
                            ]
                        ),
                        "Usability (%)": entity_summary["usability"][
                            "composite_pct"
                        ],
                        "Priority": entity_summary["schedule"]["recommendation"],
                    }
                )

        portfolio_table = pd.DataFrame(portfolio_rows)
        if not portfolio_table.empty:
            portfolio_table = portfolio_table.sort_values(
                ["Risk score", "RUL (h)"], ascending=[False, True]
            ).reset_index(drop=True)
        high_priority_count = (
            int((portfolio_table["Priority"] == "Intervene now").sum())
            if not portfolio_table.empty
            else 0
        )
        next_shift_count = (
            int((portfolio_table["Priority"] == "Plan next shift").sum())
            if not portfolio_table.empty
            else 0
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
            ds_x, ds_y = _downsample_for_plot(asset_df[time_col], asset_df[col])
            trend_fig.add_trace(
                go.Scatter(
                    x=ds_x,
                    y=ds_y,
                    name=sensor_labels.get(col, col),
                    line={"width": 2, "color": color},
                ),
                row=1,
                col=1,
            )

        ds_rx, ds_ry = _downsample_for_plot(
            asset_df[time_col], asset_df["risk_score"]
        )
        trend_fig.add_trace(
            go.Scatter(
                x=ds_rx,
                y=ds_ry,
                name="Risk score",
                line={"width": 3, "color": "#b45309"},
                fill="tozeroy",
                fillcolor="rgba(180, 83, 9, 0.12)",
            ),
            row=2,
            col=1,
        )
        trend_fig.add_hrect(
            y0=rul_info["risk_threshold"],
            y1=100,
            fillcolor="rgba(185, 28, 28, 0.12)",
            line_width=0,
            row=2,
            col=1,
        )
        apply_panel_layout(
            trend_fig,
            title="Telemetry and maintenance risk",
            height=540,
            top_margin=82,
            legend_y=1.03,
        )
        trend_fig.update_yaxes(title_text="Telemetry", row=1, col=1)
        trend_fig.update_yaxes(
            title_text="Risk score", row=2, col=1, range=[0, 100]
        )

        driver_fig = px.bar(
            driver_df,
            x="magnitude",
            y="signal",
            orientation="h",
            color="magnitude",
            color_continuous_scale=["#dbe5ea", "#315c72", "#b45309"],
            title="Current maintenance drivers",
        )
        apply_panel_layout(
            driver_fig,
            title="Current maintenance drivers",
            height=300,
            top_margin=68,
            show_legend=False,
        )
        driver_fig.update_layout(coloraxis_showscale=False)
        driver_fig.update_yaxes(categoryorder="total ascending")

        action_table = pd.DataFrame(
            [
                {
                    "Decision": schedule["recommendation"],
                    "Window": schedule["maintenance_window"],
                    "RUL (h)": rul_info["composite_rul_hours"],
                    "Usability (%)": usability["composite_pct"],
                    "Optimal maint. (h)": schedule["optimal_hour"],
                    "Warning horizon (h)": warning_profile[
                        "warning_horizon_hours"
                    ],
                }
            ]
        )

        deg_fig = _build_degradation_fig(
            asset_df, degradation, sensor_cols, time_col, sensor_labels
        )
        sched_fig = _build_schedule_fig(schedule)
        usability_fig = _build_usability_fig(usability, sensor_labels)
        warning_fig = _build_warning_timeline_fig(warning_profile)
        progression_fig = _build_failure_progression_fig(precursor_df)
        regime_fig = _build_regime_fig(regime_summary)
        portfolio_fig = (
            _build_portfolio_fig(portfolio_table)
            if not portfolio_table.empty
            else go.Figure()
        )

        economics_table = pd.DataFrame(
            [
                {
                    "Scenario": "Intervene now",
                    "Timing (h)": 0,
                    "Planned downtime": 34,
                    "Failure exposure": max(6, int(summary["risk_score"] * 0.18)),
                    "Avoidable disruption": 26,
                },
                {
                    "Scenario": "Optimal window",
                    "Timing (h)": schedule["optimal_hour"],
                    "Planned downtime": 20,
                    "Failure exposure": int(schedule["failure_at_optimal"] * 100),
                    "Avoidable disruption": 10,
                },
                {
                    "Scenario": "Defer beyond runway",
                    "Timing (h)": int(
                        max(
                            schedule["optimal_hour"] + 24,
                            rul_info["composite_rul_hours"],
                        )
                    ),
                    "Planned downtime": 12,
                    "Failure exposure": min(
                        95,
                        int(
                            max(
                                schedule["failure_at_optimal"] * 155,
                                summary["risk_score"] * 0.9,
                            )
                        ),
                    ),
                    "Avoidable disruption": 34,
                },
            ]
        )
        economics_table["Total cost index"] = economics_table[
            ["Planned downtime", "Failure exposure", "Avoidable disruption"]
        ].sum(axis=1)
        economics_table["Failure risk (%)"] = economics_table[
            "Failure exposure"
        ].map(lambda v: f"{v:.0f}%")
        economics_fig = _build_economics_fig(economics_table)

        incident_card = {
            "title": sensor_labels.get(
                rul_info["weakest_sensor"], rul_info["weakest_sensor"]
            ),
            "state": warning_profile["state"],
            "what_changed": incident_summary["what_changed"],
            "where": incident_summary["where"],
            "consequence": incident_summary["consequence"],
            "recommended_action": incident_summary["recommended_action"],
        }

        return {
            "asset_df": asset_df,
            "trend_fig": trend_fig,
            "driver_fig": driver_fig,
            "action_table": action_table,
            "risk_score": float(latest["risk_score"]),
            "health_score": float(latest["health_score"]),
            "availability_proxy": float(usability["composite_pct"]),
            "runway_hours": rul_info["composite_rul_hours"],
            "recommendation": schedule["recommendation"],
            "maintenance_window": schedule["maintenance_window"],
            "driver_df": driver_df,
            "degradation": degradation,
            "rul_info": rul_info,
            "schedule": schedule,
            "usability": usability,
            "deg_fig": deg_fig,
            "sched_fig": sched_fig,
            "usability_fig": usability_fig,
            "sensor_labels": sensor_labels,
            "warning_profile": warning_profile,
            "warning_fig": warning_fig,
            "precursor_df": precursor_df,
            "progression_fig": progression_fig,
            "regime_summary": regime_summary,
            "regime_fig": regime_fig,
            "portfolio_table": portfolio_table,
            "portfolio_fig": portfolio_fig,
            "portfolio_note": portfolio_note,
            "high_priority_count": high_priority_count,
            "next_shift_count": next_shift_count,
            "economics_table": economics_table,
            "economics_fig": economics_fig,
            "incident_card": incident_card,
        }


    def generate_demo_ev_data():
        rng = np.random.default_rng(21)
        suppliers = ["Supplier-A", "Supplier-B", "Supplier-C"]
        lines = ["Line-1", "Line-2", "Line-3"]
        shifts = ["Day", "Swing", "Night"]
        dealers = [
            ("Zeus Austin", "US-South"),
            ("Zeus Fremont", "US-West"),
            ("Zeus Phoenix", "US-West"),
            ("Zeus Newark", "US-East"),
            ("Zeus Chicago", "US-Central"),
        ]
        batches = [f"BATCH-{idx:03d}" for idx in range(1, 61)]
        rows = []

        for batch in batches:
            batch_num = int(batch.split("-")[-1])
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
                vehicle_num = cell_idx // 4 + 1
                dealer_name, market = dealers[
                    (
                        batch_num
                        + vehicle_num
                        + lines.index(line)
                        + shifts.index(shift)
                    )
                    % len(dealers)
                ]
                sale_seed = (
                    vehicle_num
                    + batch_num
                    + lines.index(line) * 2
                    + shifts.index(shift)
                )
                sale_status = "unsold" if sale_seed % 5 < 2 else "sold"
                action_priority = (
                    "P1 hold" if sale_status == "unsold" else "P1 recall"
                )

                ambient = rng.normal(24.8 + temp_shift, 1.8)
                overhang = rng.normal(1.05 + line_bias * 0.1, 0.08)
                electrolyte = rng.normal(4.25 - supplier_bias * 0.12, 0.12)
                resistance = rng.normal(
                    38 + supplier_bias * 3 + line_bias * 1.5, 2.4
                )
                capacity = rng.normal(
                    2995 - supplier_bias * 45 - line_bias * 20, 35
                )
                retention = rng.normal(
                    93.5 - supplier_bias * 1.5 - line_bias * 1.2, 1.4
                )

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
                        "Lot_ID": batch.replace("BATCH", "LOT"),
                        "Vehicle_ID": f"{batch}-VIN-{vehicle_num:03d}",
                        "Dealer_ID": dealer_name,
                        "Market": market,
                        "Sale_Status": sale_status,
                        "Action_Status": "Awaiting containment decision",
                        "Action_Priority": action_priority,
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


    def ensure_ev_traceability_fields(
        df, batch_col, supplier_col, line_col, shift_col
    ):
        df = df.copy()
        batch_series = df[batch_col].astype(str)
        row_order = df.groupby(batch_col).cumcount()
        vehicle_num = (row_order // 4) + 1

        if "Lot_ID" in df.columns:
            df["Lot_ID"] = df["Lot_ID"].astype(str)
        else:
            df["Lot_ID"] = batch_series.str.replace("BATCH", "LOT", regex=False)

        if "Vehicle_ID" in df.columns:
            df["Vehicle_ID"] = df["Vehicle_ID"].astype(str)
        else:
            df["Vehicle_ID"] = (
                batch_series + "-VIN-" + vehicle_num.astype(str).str.zfill(3)
            )

        dealer_catalog = pd.DataFrame(
            [
                {"Dealer_ID": "Zeus Austin", "Market": "US-South"},
                {"Dealer_ID": "Zeus Fremont", "Market": "US-West"},
                {"Dealer_ID": "Zeus Phoenix", "Market": "US-West"},
                {"Dealer_ID": "Zeus Newark", "Market": "US-East"},
                {"Dealer_ID": "Zeus Chicago", "Market": "US-Central"},
            ]
        )
        supplier_code = pd.factorize(df[supplier_col].astype(str))[0]
        line_code = pd.factorize(df[line_col].astype(str))[0]
        shift_code = pd.factorize(df[shift_col].astype(str))[0]
        dealer_idx = (
            vehicle_num.to_numpy() + supplier_code * 2 + line_code + shift_code
        ) % len(dealer_catalog)

        if "Dealer_ID" in df.columns:
            df["Dealer_ID"] = df["Dealer_ID"].astype(str)
        else:
            df["Dealer_ID"] = dealer_catalog.iloc[dealer_idx][
                "Dealer_ID"
            ].to_numpy()

        if "Market" in df.columns:
            df["Market"] = df["Market"].astype(str)
        else:
            df["Market"] = dealer_catalog.iloc[dealer_idx]["Market"].to_numpy()

        if "Sale_Status" in df.columns:
            normalized_status = (
                df["Sale_Status"].astype(str).str.strip().str.lower()
            )
            df["Sale_Status"] = np.where(
                normalized_status.isin(
                    ["unsold", "inventory", "dealer", "in_stock"]
                ),
                "unsold",
                "sold",
            )
        else:
            sale_seed = vehicle_num.to_numpy() + line_code * 2 + shift_code
            df["Sale_Status"] = np.where(sale_seed % 5 < 2, "unsold", "sold")

        if "Action_Status" in df.columns:
            df["Action_Status"] = df["Action_Status"].astype(str)
        else:
            df["Action_Status"] = "Awaiting containment decision"

        if "Action_Priority" in df.columns:
            df["Action_Priority"] = df["Action_Priority"].astype(str)
        else:
            df["Action_Priority"] = np.where(
                df["Sale_Status"].eq("unsold"), "P1 hold", "P1 recall"
            )

        return df


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

        df = ensure_ev_traceability_fields(
            df,
            batch_col=required["batch"],
            supplier_col=required["supplier"],
            line_col=required["line"],
            shift_col=required["shift"],
        )

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
            "lot_col": "Lot_ID",
            "vehicle_col": "Vehicle_ID",
            "dealer_col": "Dealer_ID",
            "sale_status_col": "Sale_Status",
            "market_col": "Market",
            "action_status_col": "Action_Status",
            "action_priority_col": "Action_Priority",
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
                                OneHotEncoder(
                                    handle_unknown="ignore", sparse_output=False
                                ),
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
            "average_precision": float(
                average_precision_score(y_test, test_probs)
            ),
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
        importance_df["feature"] = (
            importance_df["feature"]
            .str.replace("numeric__", "", regex=False)
            .str.replace("categorical__encoder__", "", regex=False)
        )

        return df, metrics, importance_df


    def lineage_figure(vehicle_actions, lot_col):
        if vehicle_actions.empty:
            return go.Figure()

        flow = vehicle_actions[
            ["supplier", "line", "shift", lot_col, "containment_cluster"]
        ].copy()
        layer_specs = [
            ("supplier", "Supplier", "#315c72"),
            ("line", "Line", "#8aa1af"),
            ("shift", "Shift", "#cf8f3d"),
            (lot_col, "Lot", "#475569"),
            ("containment_cluster", "Action", None),
        ]

        node_index = {}
        labels = []
        colors = []

        def _cluster_color(value):
            return "#cf8f3d" if "Unsold" in value else "#b91c1c"

        for column, title, color in layer_specs:
            values = flow[column].astype(str).drop_duplicates().tolist()
            for value in values:
                node_index[(column, value)] = len(labels)
                labels.append(title + "\n" + value)
                colors.append(color or _cluster_color(value))

        sources = []
        targets = []
        values = []
        link_colors = []
        transitions = [
            ("supplier", "line"),
            ("line", "shift"),
            ("shift", lot_col),
            (lot_col, "containment_cluster"),
        ]
        for source_col, target_col in transitions:
            grouped = (
                flow.groupby([source_col, target_col])
                .size()
                .reset_index(name="count")
            )
            for _, row in grouped.iterrows():
                target_value = str(row[target_col])
                sources.append(node_index[(source_col, str(row[source_col]))])
                targets.append(node_index[(target_col, target_value)])
                values.append(int(row["count"]))
                if target_col == "containment_cluster":
                    link_colors.append(
                        "rgba(207, 143, 61, 0.35)"
                        if "Unsold" in target_value
                        else "rgba(185, 28, 28, 0.30)"
                    )
                else:
                    link_colors.append("rgba(49, 92, 114, 0.16)")

        fig = go.Figure(
            go.Sankey(
                arrangement="snap",
                node={
                    "pad": 20,
                    "thickness": 18,
                    "line": {"color": "rgba(255,255,255,0.9)", "width": 1},
                    "label": labels,
                    "color": colors,
                },
                link={
                    "source": sources,
                    "target": targets,
                    "value": values,
                    "color": link_colors,
                },
            )
        )
        apply_panel_layout(
            fig,
            title="Lot containment flow",
            height=380,
            top_margin=74,
            show_legend=False,
        )
        fig.update_layout(font={"size": 13, "color": "#13212b"})
        return fig


    def impact_split_figure(containment_actions):
        if containment_actions.empty:
            return go.Figure()

        values = containment_actions["Vehicle count"].astype(int).tolist()
        labels = containment_actions["Cluster"].tolist()
        colors = ["#cf8f3d", "#b91c1c"]
        total = max(sum(values), 1)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=["Impacted vehicles"],
                x=[values[0]],
                orientation="h",
                name=labels[0],
                marker={"color": colors[0]},
                text=[f"{values[0]} hold"],
                textposition="inside",
                insidetextanchor="middle",
            )
        )
        fig.add_trace(
            go.Bar(
                y=["Impacted vehicles"],
                x=[values[1]],
                orientation="h",
                name=labels[1],
                marker={"color": colors[1]},
                text=[f"{values[1]} recall"],
                textposition="inside",
                insidetextanchor="middle",
            )
        )
        apply_panel_layout(
            fig,
            title="Downstream impact split",
            height=380,
            top_margin=72,
            bottom_margin=20,
            show_legend=False,
        )
        fig.update_layout(barmode="stack")
        fig.update_xaxes(title_text="Vehicle count", range=[0, total])
        fig.add_annotation(
            x=total,
            y=0,
            text=f"Total impacted: {total}",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            yshift=16,
            font={"size": 13, "color": "#13212b"},
        )
        return fig


    def _build_process_drift_fig(modeled_df, supplier_col, line_col, shift_col):
        grouped = (
            modeled_df.groupby([supplier_col, line_col, shift_col])
            .agg(avg_risk=("predicted_risk", "mean"))
            .reset_index()
        )
        grouped["Line / Shift"] = (
            grouped[line_col].astype(str) + " / " + grouped[shift_col].astype(str)
        )
        pivot = grouped.pivot_table(
            index=supplier_col,
            columns="Line / Shift",
            values="avg_risk",
            fill_value=0,
        )
        fig = px.imshow(
            pivot,
            color_continuous_scale=["#dbe5ea", "#8aa1af", "#cf8f3d", "#b91c1c"],
            aspect="auto",
            title="Process drift by supplier / line / shift",
        )
        apply_panel_layout(
            fig,
            title="Process drift by supplier / line / shift",
            height=360,
            top_margin=74,
            show_legend=False,
        )
        fig.update_layout(coloraxis_colorbar_title="Avg risk")
        return fig


    def _build_quality_fingerprint_fig(lead_indicator_table):
        if lead_indicator_table.empty:
            return go.Figure()
        fig = px.bar(
            lead_indicator_table.sort_values(
                "Delta vs healthy (z)", ascending=True
            ),
            x="Delta vs healthy (z)",
            y="Signal",
            orientation="h",
            color="Direction",
            color_discrete_map={
                "Higher than healthy": "#b91c1c",
                "Lower than healthy": "#315c72",
            },
            title="Suspect lot fingerprint vs healthy baseline",
        )
        apply_panel_layout(
            fig,
            title="Suspect lot fingerprint vs healthy baseline",
            height=340,
            top_margin=74,
            legend_y=1.03,
        )
        return fig


    def _build_quality_control_fig(batch_summary, batch_col, focus_batch):
        chart_df = batch_summary.sort_values(batch_col).copy()
        center = float(chart_df["avg_risk"].mean())
        upper = float(chart_df["avg_risk"].mean() + chart_df["avg_risk"].std())
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=chart_df[batch_col],
                y=chart_df["avg_risk"],
                mode="lines+markers",
                name="Avg batch risk",
                line={"width": 2, "color": "#315c72"},
            )
        )
        focus_row = chart_df.loc[
            chart_df[batch_col].astype(str) == str(focus_batch)
        ]
        if not focus_row.empty:
            fig.add_trace(
                go.Scatter(
                    x=focus_row[batch_col],
                    y=focus_row["avg_risk"],
                    mode="markers",
                    name="Focus lot",
                    marker={
                        "size": 14,
                        "color": "#b91c1c",
                        "line": {"color": "#ffffff", "width": 1.5},
                    },
                )
            )
        fig.add_hline(
            y=center,
            line_dash="dash",
            line_color="#6b7280",
            annotation_text="Center line",
        )
        fig.add_hline(
            y=upper,
            line_dash="dot",
            line_color="#cf8f3d",
            annotation_text="Early warning band",
        )
        apply_panel_layout(
            fig,
            title="Batch quality control chart",
            height=360,
            top_margin=74,
            legend_y=1.03,
        )
        fig.update_layout(xaxis_title="Batch", yaxis_title="Avg predicted risk")
        return fig


    def analyze_ev(ev_state, selected_supplier, selected_line, selected_batch):
        modeled_df, metrics, importance_df = _ev_risk_model(ev_state)
        supplier_col = ev_state["supplier_col"]
        line_col = ev_state["line_col"]
        shift_col = ev_state["shift_col"]
        batch_col = ev_state["batch_col"]
        grade_col = ev_state["grade_col"]
        defect_col = ev_state["defect_col"]
        lot_col = ev_state["lot_col"]
        vehicle_col = ev_state["vehicle_col"]
        dealer_col = ev_state["dealer_col"]
        sale_status_col = ev_state["sale_status_col"]
        market_col = ev_state["market_col"]

        filtered = modeled_df.copy()
        if selected_supplier != "All":
            filtered = filtered.loc[
                filtered[supplier_col].astype(str) == selected_supplier
            ]
        if selected_line != "All":
            filtered = filtered.loc[
                filtered[line_col].astype(str) == selected_line
            ]
        if selected_batch != "All":
            filtered = filtered.loc[
                filtered[batch_col].astype(str) == selected_batch
            ]
        if filtered.empty:
            filtered = modeled_df.copy()

        def _cluster_vehicle_count(values, target_status):
            rows = filtered.loc[
                values.index, [vehicle_col, sale_status_col]
            ].copy()
            status = rows[sale_status_col].astype(str).str.lower()
            return int(
                rows.loc[status == target_status, vehicle_col]
                .astype(str)
                .nunique()
            )

        def _worst_grade(values):
            rank = {"Grade A": 0, "Grade B": 1, "Scrap": 2}
            values = values.astype(str)
            return max(values, key=lambda value: rank.get(value, 0))

        def _dominant_defect(values):
            cleaned = values.astype(str).replace("None", np.nan).dropna()
            if cleaned.empty:
                return "No defect observed"
            return cleaned.value_counts().index[0]

        qc_mix = (
            filtered.groupby(grade_col)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        supplier_hotspots = (
            modeled_df.groupby(supplier_col)
            .agg(
                affected_share=(
                    grade_col,
                    lambda values: (values != "Grade A").mean(),
                ),
                avg_risk=("predicted_risk", "mean"),
            )
            .reset_index()
            .sort_values("avg_risk", ascending=False)
        )
        batch_summary = (
            filtered.groupby(batch_col)
            .agg(
                lot_id=(lot_col, "first"),
                supplier=(supplier_col, "first"),
                line=(line_col, "first"),
                shift=(shift_col, "first"),
                avg_risk=("predicted_risk", "mean"),
                scrap_share=(grade_col, lambda values: (values == "Scrap").mean()),
                impacted_vehicle_count=(
                    vehicle_col,
                    lambda values: values.astype(str).nunique(),
                ),
                dealer_hold_count=(
                    vehicle_col,
                    lambda values: _cluster_vehicle_count(values, "unsold"),
                ),
                recall_count=(
                    vehicle_col,
                    lambda values: _cluster_vehicle_count(values, "sold"),
                ),
            )
            .reset_index()
            .sort_values(["avg_risk", "scrap_share"], ascending=False)
        )

        focus_batch = selected_batch
        if focus_batch == "All":
            focus_batch = str(batch_summary.iloc[0][batch_col])
        focus_rows = filtered.loc[
            filtered[batch_col].astype(str) == focus_batch
        ].copy()

        vehicle_actions = (
            focus_rows.groupby(
                [lot_col, vehicle_col, dealer_col, market_col, sale_status_col],
                as_index=False,
            )
            .agg(
                supplier=(supplier_col, "first"),
                line=(line_col, "first"),
                shift=(shift_col, "first"),
                avg_risk=("predicted_risk", "mean"),
                peak_cell_risk=("predicted_risk", "max"),
                worst_grade=(grade_col, _worst_grade),
                dominant_defect=(defect_col, _dominant_defect),
            )
            .sort_values("avg_risk", ascending=False)
        )
        vehicle_actions[sale_status_col] = (
            vehicle_actions[sale_status_col].astype(str).str.lower()
        )
        vehicle_actions["containment_cluster"] = np.where(
            vehicle_actions[sale_status_col].eq("unsold"),
            "Unsold dealer inventory",
            "Sold vehicles requiring recall",
        )
        vehicle_actions["recommended_action"] = np.where(
            vehicle_actions[sale_status_col].eq("unsold"),
            "Place dealer hold and quarantine inventory",
            "Issue recall and route vehicle to urgent inspection",
        )
        vehicle_actions["Action_Status"] = np.where(
            vehicle_actions[sale_status_col].eq("unsold"),
            "Ready to notify dealer operations",
            "Ready to notify owners and field service",
        )
        vehicle_actions["Action_Priority"] = np.where(
            vehicle_actions[sale_status_col].eq("unsold"), "P1 hold", "P1 recall"
        )
        vehicle_actions["avg_risk"] = vehicle_actions["avg_risk"].round(2)
        vehicle_actions["peak_cell_risk"] = vehicle_actions[
            "peak_cell_risk"
        ].round(2)

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
        apply_panel_layout(
            qc_fig,
            title="Quality mix for current filters",
            height=320,
            top_margin=70,
            bottom_margin=20,
            legend_y=1.03,
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
        apply_panel_layout(
            hotspot_fig,
            title="Supplier genealogy hotspots",
            height=320,
            top_margin=70,
            bottom_margin=20,
            show_legend=False,
        )
        hotspot_fig.update_layout(coloraxis_colorbar_title="Affected share")

        importance_fig = px.bar(
            importance_df.sort_values("importance", ascending=True),
            x="importance",
            y="feature",
            orientation="h",
            title="Batch quality risk drivers",
            color="importance",
            color_continuous_scale=["#dbe5ea", "#315c72", "#b45309"],
        )
        apply_panel_layout(
            importance_fig,
            title="Batch quality risk drivers",
            height=340,
            top_margin=70,
            bottom_margin=20,
            show_legend=False,
        )
        importance_fig.update_layout(coloraxis_showscale=False)

        lineage_fig = lineage_figure(vehicle_actions, lot_col)
        impact_fig = impact_split_figure(
            pd.DataFrame(
                [
                    {
                        "Cluster": "Unsold vehicles on dealer lots",
                        "Vehicle count": int(
                            vehicle_actions.loc[
                                vehicle_actions[sale_status_col].eq("unsold"),
                                vehicle_col,
                            ].nunique()
                        ),
                    },
                    {
                        "Cluster": "Sold vehicles requiring recall",
                        "Vehicle count": int(
                            vehicle_actions.loc[
                                vehicle_actions[sale_status_col].eq("sold"),
                                vehicle_col,
                            ].nunique()
                        ),
                    },
                ]
            )
        )

        defect_view = (
            focus_rows.groupby(defect_col)
            .size()
            .reset_index(name="Affected cells")
            .sort_values("Affected cells", ascending=False)
            .head(5)
        )
        defect_table = defect_view.rename(columns={defect_col: "Defect"})

        affected_lots = (
            batch_summary.head(5)
            .rename(
                columns={
                    batch_col: "Batch",
                    "lot_id": "Lot",
                    "supplier": "Supplier",
                    "line": "Line",
                    "shift": "Shift",
                    "avg_risk": "Avg risk",
                    "scrap_share": "Scrap share",
                    "impacted_vehicle_count": "Vehicles",
                    "dealer_hold_count": "Dealer hold",
                    "recall_count": "Recall",
                }
            )
            .copy()
        )
        affected_lots["Avg risk"] = affected_lots["Avg risk"].round(2)
        affected_lots["Scrap share"] = (
            affected_lots["Scrap share"]
            .mul(100)
            .round(1)
            .map(lambda value: f"{value:.1f}%")
        )

        containment_actions = pd.DataFrame(
            [
                {
                    "Cluster": "Unsold vehicles on dealer lots",
                    "Vehicle count": int(
                        vehicle_actions.loc[
                            vehicle_actions[sale_status_col].eq("unsold"),
                            vehicle_col,
                        ].nunique()
                    ),
                    "Recommended action": "Send dealer hold notice and quarantine inventory",
                    "Owner": "Dealer operations",
                    "Priority": "P1 hold",
                },
                {
                    "Cluster": "Sold vehicles requiring recall",
                    "Vehicle count": int(
                        vehicle_actions.loc[
                            vehicle_actions[sale_status_col].eq("sold"),
                            vehicle_col,
                        ].nunique()
                    ),
                    "Recommended action": "Launch recall / urgent inspection campaign",
                    "Owner": "Field service + customer care",
                    "Priority": "P1 recall",
                },
            ]
        )

        action_queue = (
            vehicle_actions[
                [
                    "containment_cluster",
                    "Action_Priority",
                    vehicle_col,
                    dealer_col,
                    market_col,
                    "avg_risk",
                    "worst_grade",
                    "dominant_defect",
                    "recommended_action",
                    "Action_Status",
                ]
            ]
            .rename(
                columns={
                    "containment_cluster": "Cluster",
                    vehicle_col: "Vehicle ID",
                    dealer_col: "Dealer",
                    market_col: "Market",
                    "avg_risk": "Avg risk",
                    "worst_grade": "Worst grade",
                    "dominant_defect": "Likely defect",
                    "recommended_action": "Action",
                    "Action_Status": "Action status",
                    "Action_Priority": "Priority",
                }
            )
            .copy()
        )
        action_queue["cluster_order"] = np.where(
            action_queue["Cluster"].eq("Unsold dealer inventory"), 0, 1
        )
        action_queue = action_queue.sort_values(
            ["cluster_order", "Avg risk"], ascending=[True, False]
        ).drop(columns=["cluster_order"])

        dominant_defect = (
            str(defect_table.iloc[0]["Defect"])
            if not defect_table.empty
            else "No defect observed"
        )
        worst_grade = (
            str(vehicle_actions["worst_grade"].iloc[0])
            if not vehicle_actions.empty
            else "Grade A"
        )
        focus_supplier = str(focus_rows[supplier_col].iloc[0])
        focus_line = str(focus_rows[line_col].iloc[0])
        focus_shift = str(focus_rows[shift_col].iloc[0])
        focus_lot = str(focus_rows[lot_col].iloc[0])
        containment_summary = {
            "affected_lot": focus_lot,
            "supplier": focus_supplier,
            "line": focus_line,
            "shift": focus_shift,
            "impacted_vehicle_count": int(vehicle_actions.shape[0]),
            "dealer_hold_count": int(containment_actions.iloc[0]["Vehicle count"]),
            "recall_count": int(containment_actions.iloc[1]["Vehicle count"]),
            "avg_risk": round(float(focus_rows["predicted_risk"].mean()), 2),
            "dominant_defect": dominant_defect,
            "worst_grade": worst_grade,
            "business_action": "Hold unsold dealer inventory now and issue urgent recall / inspection for sold vehicles.",
        }

        feature_candidates = [
            "Ambient_Temp_C",
            "Anode_Overhang_mm",
            "Electrolyte_Volume_ml",
            "Internal_Resistance_mOhm",
            "Capacity_mAh",
            "Retention_50Cycle_Pct",
        ]
        quality_features = [
            col for col in feature_candidates if col in modeled_df.columns
        ]
        healthy_baseline = modeled_df.loc[
            (modeled_df[grade_col].astype(str) == "Grade A")
            & (
                modeled_df["predicted_risk"]
                <= modeled_df["predicted_risk"].quantile(0.35)
            ),
            quality_features,
        ].copy()
        if healthy_baseline.empty:
            healthy_baseline = modeled_df[quality_features].copy()
        focus_signature = focus_rows[quality_features].mean(numeric_only=True)
        baseline_mean = healthy_baseline.mean(numeric_only=True)
        baseline_std = (
            healthy_baseline.std(numeric_only=True).replace(0, np.nan).fillna(1.0)
        )
        lead_indicator_table = pd.DataFrame(
            {
                "Signal": [
                    col.replace("_", " ").replace("Pct", "%").title()
                    for col in quality_features
                ],
                "Delta vs healthy (z)": (
                    (focus_signature - baseline_mean) / baseline_std
                ).values,
            }
        )
        lead_indicator_table["Direction"] = np.where(
            lead_indicator_table["Delta vs healthy (z)"] >= 0,
            "Higher than healthy",
            "Lower than healthy",
        )
        lead_indicator_table["Delta vs healthy (z)"] = lead_indicator_table[
            "Delta vs healthy (z)"
        ].round(2)
        lead_indicator_table = lead_indicator_table.reindex(
            lead_indicator_table["Delta vs healthy (z)"]
            .abs()
            .sort_values(ascending=False)
            .index
        ).reset_index(drop=True)

        process_drift_fig = _build_process_drift_fig(
            modeled_df, supplier_col, line_col, shift_col
        )
        fingerprint_fig = _build_quality_fingerprint_fig(
            lead_indicator_table.head(6)
        )
        quality_control_fig = _build_quality_control_fig(
            batch_summary, batch_col, focus_batch
        )

        predictive_quality_summary = pd.DataFrame(
            [
                {
                    "View": "Process drift",
                    "Takeaway": f"{supplier_hotspots.iloc[0][supplier_col]} is the highest-risk supplier pattern in the current data.",
                },
                {
                    "View": "Fingerprint",
                    "Takeaway": f"{lead_indicator_table.iloc[0]['Signal']} is the largest deviation from the healthy process baseline.",
                },
                {
                    "View": "Control chart",
                    "Takeaway": f"Batch {focus_batch} is highlighted as the current focus lot for quality escalation.",
                },
            ]
        )

        return {
            "modeled_df": modeled_df,
            "metrics": metrics,
            "qc_fig": qc_fig,
            "hotspot_fig": hotspot_fig,
            "importance_fig": importance_fig,
            "lineage_fig": lineage_fig,
            "impact_fig": impact_fig,
            "batch_summary": batch_summary,
            "top_batch": batch_summary.iloc[0],
            "focus_batch": focus_batch,
            "defect_table": defect_table,
            "affected_lots": affected_lots,
            "containment_actions": containment_actions,
            "action_queue": action_queue,
            "containment_summary": containment_summary,
            "process_drift_fig": process_drift_fig,
            "fingerprint_fig": fingerprint_fig,
            "quality_control_fig": quality_control_fig,
            "lead_indicator_table": lead_indicator_table,
            "predictive_quality_summary": predictive_quality_summary,
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
            Enterprise Traceability + Predictive Maintenance
          </div>
          <h1 style="margin:0.35rem 0 0; font-size:2rem;">
            Operational command center for maintenance and containment
          </h1>
          <p>
            Identify the affected lot, trace downstream exposure, separate unsold from sold units,
            and trigger the right field action alongside maintenance decisions. Refreshed on
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
            - Demo mode now simulates lot-to-vehicle, dealer, and sale-status mappings so the containment workflow stays reviewable before downstream system onboarding.
            - This keeps the hold-versus-recall story visible even when real inventory and service data are not yet connected.
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

    batches = ["All"] + sorted(
        filtered[ev_state["batch_col"]].astype(str).unique().tolist()
    )
    batch_picker = mo.ui.dropdown(
        batches,
        value="All",
        label="selected_batch",
        full_width=True,
    )
    mo.hstack(
        [
            mo.md("### Containment focus"),
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
                value=f"{metro_analysis['warning_profile']['warning_horizon_hours']}h",
                label="early warning horizon",
                caption=metro_analysis["warning_profile"]["state"],
                bordered=True,
            ),
            mo.stat(
                value=f"{metro_analysis['rul_info']['composite_rul_hours']}h",
                label="remaining useful life",
                caption=f"optimal maint. at {metro_analysis['schedule']['optimal_hour']}h",
                bordered=True,
            ),
            mo.stat(
                value=str(metro_analysis["high_priority_count"]),
                label="priority maintenance items",
                caption=f"{metro_analysis['next_shift_count']} plan-next-shift items",
                bordered=True,
            ),
            mo.stat(
                value=ev_analysis["containment_summary"]["affected_lot"],
                label="active quality lot",
                caption="enterprise proof point",
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

        - [MetroPT-3 starter](https://www.kaggle.com/code/joebeachcapital/metropt-3-data-import-eda-starter): time-series framing and telemetry-first storytelling.
        - [Predictive maintenance and XAI](https://www.kaggle.com/code/chinmayadatt/notebook-predictive-maintenance-and-xai/notebook): maintenance-driver and explainability framing.
        - [EV Battery QC code gallery](https://www.kaggle.com/datasets/kanchana1990/ev-battery-qc-synthetic-defect-dataset/code): lot-level storytelling and QC drilldowns.

        Data placement details: [`data/README.md`]({ROOT / "data" / "README.md"}).
        """
    )
    inspiration
    return


@app.cell(hide_code=True)
def _(ev_analysis, metro_analysis, mo):
    maintenance_summary = mo.Html(
        f"""
        <style>
          .pm-maint-grid {{
            display: grid;
            grid-template-columns: minmax(280px, 1.6fr) repeat(4, minmax(130px, 1fr));
            gap: 12px;
            margin-bottom: 0.2rem;
          }}
          .pm-maint-hero, .pm-maint-stat {{
            border-radius: 18px;
            border: 1px solid #dbe3e8;
            background: #ffffff;
            box-shadow: 0 10px 22px rgba(49, 92, 114, 0.08);
          }}
          .pm-maint-hero {{
            padding: 16px 18px;
            background: linear-gradient(135deg, #13212b 0%, #315c72 62%, #8aa1af 100%);
            color: #ffffff;
          }}
          .pm-maint-hero .eyebrow {{
            font-size: 0.76rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            opacity: 0.82;
          }}
          .pm-maint-hero h3 {{
            margin: 0.35rem 0 0.25rem;
            font-size: 1.45rem;
          }}
          .pm-maint-hero p {{
            margin: 0.2rem 0;
            line-height: 1.35;
            color: rgba(255,255,255,0.88);
          }}
          .pm-maint-stat {{
            padding: 14px 16px;
          }}
          .pm-maint-stat .label {{
            color: #5a6b76;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
          }}
          .pm-maint-stat .value {{
            margin-top: 0.35rem;
            font-size: 1.45rem;
            font-weight: 700;
            color: #13212b;
          }}
          .pm-maint-stat .caption {{
            margin-top: 0.15rem;
            color: #5a6b76;
            font-size: 0.92rem;
          }}
          @media (max-width: 1100px) {{
            .pm-maint-grid {{
              grid-template-columns: repeat(2, minmax(180px, 1fr));
            }}
          }}
        </style>
        <div class="pm-maint-grid">
          <div class="pm-maint-hero">
            <div class="eyebrow">Selected maintenance case</div>
            <h3>{metro_analysis["incident_card"]["state"]}</h3>
            <p>{metro_analysis["incident_card"]["what_changed"]}</p>
            <p>{metro_analysis["incident_card"]["consequence"]}</p>
            <p>{metro_analysis["incident_card"]["recommended_action"]}</p>
          </div>
          <div class="pm-maint-stat">
            <div class="label">Warning horizon</div>
            <div class="value">{metro_analysis["warning_profile"]["warning_horizon_hours"]}h</div>
            <div class="caption">Lead time before warning band</div>
          </div>
          <div class="pm-maint-stat">
            <div class="label">RUL</div>
            <div class="value">{metro_analysis["rul_info"]["composite_rul_hours"]}h</div>
            <div class="caption">Weakest link: {metro_analysis["sensor_labels"].get(metro_analysis["rul_info"]["weakest_sensor"], metro_analysis["rul_info"]["weakest_sensor"])}</div>
          </div>
          <div class="pm-maint-stat">
            <div class="label">Priority queue</div>
            <div class="value">{metro_analysis["high_priority_count"]}</div>
            <div class="caption">Maintenance items needing immediate attention</div>
          </div>
          <div class="pm-maint-stat">
            <div class="label">Next shift plan</div>
            <div class="value">{metro_analysis["next_shift_count"]}</div>
            <div class="caption">Items to schedule in the next shift</div>
          </div>
        </div>
        """
    )

    lot_summary = mo.Html(
        f"""
        <style>
          .pm-lot-grid {{
            display: grid;
            grid-template-columns: minmax(240px, 1.4fr) repeat(5, minmax(120px, 1fr));
            gap: 12px;
            margin-bottom: 0.2rem;
          }}
          .pm-lot-hero, .pm-lot-stat {{
            border-radius: 18px;
            border: 1px solid #dbe3e8;
            background: #ffffff;
            box-shadow: 0 10px 22px rgba(49, 92, 114, 0.08);
          }}
          .pm-lot-hero {{
            padding: 16px 18px;
            background: linear-gradient(135deg, #13212b 0%, #315c72 62%, #8aa1af 100%);
            color: #ffffff;
          }}
          .pm-lot-hero .eyebrow {{
            font-size: 0.76rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            opacity: 0.82;
          }}
          .pm-lot-hero h3 {{
            margin: 0.35rem 0 0.2rem;
            font-size: 1.45rem;
          }}
          .pm-lot-hero p {{
            margin: 0.2rem 0;
            line-height: 1.35;
            color: rgba(255,255,255,0.88);
          }}
          .pm-lot-stat {{
            padding: 14px 16px;
          }}
          .pm-lot-stat .label {{
            color: #5a6b76;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
          }}
          .pm-lot-stat .value {{
            margin-top: 0.35rem;
            font-size: 1.45rem;
            font-weight: 700;
            color: #13212b;
          }}
          .pm-lot-stat .caption {{
            margin-top: 0.15rem;
            color: #5a6b76;
            font-size: 0.92rem;
          }}
          @media (max-width: 1100px) {{
            .pm-lot-grid {{
              grid-template-columns: repeat(2, minmax(180px, 1fr));
            }}
          }}
        </style>
        <div class="pm-lot-grid">
          <div class="pm-lot-hero">
            <div class="eyebrow">Enterprise proof point</div>
            <h3>{ev_analysis["containment_summary"]["affected_lot"]}</h3>
            <p>{ev_analysis["containment_summary"]["supplier"]} / {ev_analysis["containment_summary"]["line"]} / {ev_analysis["containment_summary"]["shift"]}</p>
            <p>{ev_analysis["containment_summary"]["business_action"]}</p>
          </div>
          <div class="pm-lot-stat">
            <div class="label">Impacted vehicles</div>
            <div class="value">{ev_analysis["containment_summary"]["impacted_vehicle_count"]}</div>
            <div class="caption">Downstream population traced</div>
          </div>
          <div class="pm-lot-stat">
            <div class="label">Dealer hold</div>
            <div class="value">{ev_analysis["containment_summary"]["dealer_hold_count"]}</div>
            <div class="caption">Unsold units to quarantine</div>
          </div>
          <div class="pm-lot-stat">
            <div class="label">Recall</div>
            <div class="value">{ev_analysis["containment_summary"]["recall_count"]}</div>
            <div class="caption">Sold units needing inspection</div>
          </div>
          <div class="pm-lot-stat">
            <div class="label">Average risk</div>
            <div class="value">{ev_analysis["containment_summary"]["avg_risk"]:.2f}</div>
            <div class="caption">Predicted lot-level severity</div>
          </div>
          <div class="pm-lot-stat">
            <div class="label">Dominant defect</div>
            <div class="value" style="font-size:1.05rem; line-height:1.25;">{ev_analysis["containment_summary"]["dominant_defect"]}</div>
            <div class="caption">Worst observed grade: {ev_analysis["containment_summary"]["worst_grade"]}</div>
          </div>
        </div>
        """
    )


    def panel(title, body):
        return mo.vstack([mo.md(f"#### {title}"), body], gap=0.35)


    executive_containment_actions = ev_analysis["containment_actions"].copy()
    executive_containment_actions["Recommended action"] = (
        executive_containment_actions["Recommended action"].replace(
            {
                "Send dealer hold notice and quarantine inventory": "Dealer hold and quarantine",
                "Launch recall / urgent inspection campaign": "Recall and urgent inspection",
            }
        )
    )

    executive_view = mo.vstack(
        [
            mo.md(
                f"""
                ### Executive overview

                - **Predictive maintenance posture:** {metro_analysis["recommendation"]} for the selected asset, with an action window of **{metro_analysis["maintenance_window"]}**.
                - **Early warning:** the system is surfacing a **{metro_analysis["warning_profile"]["state"]}** signal with **{metro_analysis["warning_profile"]["warning_horizon_hours"]}h** of warning lead time.
                - **Failure progression:** the current issue is being driven first by **{metro_analysis["incident_card"]["title"]}**, then by the next-ranked precursor signals in the failure ladder.
                - **Portfolio view:** the notebook now prioritizes a maintenance backlog, not just one machine. Immediate-action items: **{metro_analysis["high_priority_count"]}**.
                - **Enterprise proof:** the same stack also drives downstream traceability and containment through lot genealogy when a process-quality issue is detected.
                """
            ),
            maintenance_summary,
            panel(
                "Maintenance actions",
                mo.ui.table(
                    metro_analysis["action_table"], selection=None, page_size=4
                ),
            ),
            panel(
                "Quality containment actions",
                mo.ui.table(
                    executive_containment_actions, selection=None, page_size=4
                ),
            ),
        ],
        gap=1.0,
    )

    fleet_view = mo.vstack(
        [
            mo.md(
                f"""
                ### Fleet maintenance command center

                This view leads with predictive maintenance. It shows the selected asset’s telemetry, the early warning runway, and the ranked maintenance queue that operations would triage first.
                {metro_analysis["portfolio_note"]}
                """
            ),
            maintenance_summary,
            mo.hstack(
                [metro_analysis["trend_fig"], metro_analysis["warning_fig"]],
                widths=[1.7, 1],
                gap=1.0,
                wrap=True,
                align="stretch",
            ),
            mo.hstack(
                [metro_analysis["portfolio_fig"], metro_analysis["regime_fig"]],
                widths=[1.4, 1],
                gap=1.0,
                wrap=True,
                align="stretch",
            ),
            panel(
                "Maintenance backlog",
                mo.ui.table(
                    metro_analysis["portfolio_table"],
                    selection=None,
                    page_size=6,
                ),
            ),
        ],
        gap=1.0,
    )

    failure_view = mo.vstack(
        [
            mo.md(
                f"""
                ### Failure progression

                This view explains what changed, which signals will hit their limits first, and why the model is recommending the current intervention window.
                """
            ),
            metro_analysis["driver_fig"],
            metro_analysis["progression_fig"],
            panel(
                "Failure precursor ladder",
                mo.ui.table(
                    metro_analysis["precursor_df"], selection=None, page_size=8
                ),
            ),
            metro_analysis["deg_fig"],
        ],
        gap=1.0,
    )

    maintenance_econ_view = mo.vstack(
        [
            mo.md(
                f"""
                ### Maintenance economics

                The economics layer is deliberately heuristic. It translates the intervention timing recommendation into a directional business tradeoff between planned downtime, failure exposure, and avoidable disruption.
                """
            ),
            mo.hstack(
                [metro_analysis["sched_fig"], metro_analysis["economics_fig"]],
                widths=[1.2, 1],
                gap=1.0,
                wrap=True,
                align="stretch",
            ),
            panel(
                "Scenario comparison",
                mo.ui.table(
                    metro_analysis["economics_table"], selection=None, page_size=4
                ),
            ),
        ],
        gap=1.0,
    )

    genealogy_view = mo.vstack(
        [
            mo.md(
                f"""
                ### Lot genealogy and containment

                Lot genealogy remains the enterprise proof point: the same platform that detects degrading equipment health can also trace process issues into concrete field actions.
                """
            ),
            lot_summary,
            mo.hstack(
                [ev_analysis["lineage_fig"], ev_analysis["impact_fig"]],
                widths=[1.8, 1],
                gap=1.0,
                wrap=True,
                align="stretch",
            ),
            panel(
                "Operational action queue",
                mo.ui.table(
                    ev_analysis["action_queue"], selection=None, page_size=8
                ),
            ),
            panel(
                "Supporting lot evidence",
                mo.ui.table(
                    ev_analysis["affected_lots"], selection=None, page_size=4
                ),
            ),
            panel(
                "Dominant defect pattern",
                mo.ui.table(
                    ev_analysis["defect_table"], selection=None, page_size=4
                ),
            ),
        ],
        gap=1.0,
    )

    predictive_quality_view = mo.vstack(
        [
            mo.md(
                f"""
                ### Predictive quality

                This is the sister capability to predictive maintenance. Instead of waiting for bad lots to surface at the end of inspection, the notebook shows which process signatures are drifting away from healthy operating conditions.
                """
            ),
            mo.hstack(
                [
                    ev_analysis["process_drift_fig"],
                    ev_analysis["quality_control_fig"],
                ],
                widths="equal",
                gap=1.0,
                wrap=True,
                align="stretch",
            ),
            mo.hstack(
                [ev_analysis["fingerprint_fig"], ev_analysis["importance_fig"]],
                widths="equal",
                gap=1.0,
                wrap=True,
                align="stretch",
            ),
            panel(
                "Lead indicators before lot failure",
                mo.ui.table(
                    ev_analysis["lead_indicator_table"],
                    selection=None,
                    page_size=6,
                ),
            ),
            panel(
                "Predictive quality summary",
                mo.ui.table(
                    ev_analysis["predictive_quality_summary"],
                    selection=None,
                    page_size=4,
                ),
            ),
        ],
        gap=1.0,
    )

    recommendations = mo.md(
        f"""
        ### Operational recommendations

        1. **Maintenance timing:** schedule maintenance at **{metro_analysis["schedule"]["optimal_hour"]}h** (window: {metro_analysis["maintenance_window"]}). Current urgency is **{metro_analysis["risk_score"]:.0f}/100**.
        2. **Early warning:** communicate that the system surfaces a **{metro_analysis["warning_profile"]["state"]}** signal with **{metro_analysis["warning_profile"]["warning_horizon_hours"]}h** of advance warning before the machine enters the intervention boundary.
        3. **Portfolio triage:** prioritize **{metro_analysis["high_priority_count"]}** immediate maintenance items first, then **{metro_analysis["next_shift_count"]}** plan-next-shift items from the backlog board.
        4. **Failure explanation:** explain the maintenance issue through the top precursor chain, starting with **{metro_analysis["incident_card"]["title"]}** and the driver ladder in the failure-progression tab.
        5. **Enterprise extension:** use the lot genealogy and predictive-quality tabs to show that the same stack can move from machine health to process health to downstream containment actions.
        """
    )

    app_tabs = mo.ui.tabs(
        {
            "Executive overview": executive_view,
            "Fleet maintenance": fleet_view,
            "Failure progression": failure_view,
            "Maintenance economics": maintenance_econ_view,
            "Lot genealogy": genealogy_view,
            "Predictive quality": predictive_quality_view,
            "Recommendations": recommendations,
        },
        value="Executive overview",
    )
    app_tabs
    return


if __name__ == "__main__":
    app.run()
