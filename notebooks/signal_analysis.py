# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "antropy==0.2.2",
#     "filterpy==1.4.5",
#     "hmmlearn==0.3.3",
#     "marimo>=0.22.4",
#     "numpy==2.4.4",
#     "pandas==3.0.2",
#     "plotly==6.6.0",
#     "PyWavelets==1.9.0",
#     "scikit-learn==1.8.0",
#     "statsmodels==0.14.6",
# ]
# ///

import marimo

__generated_with = "0.22.5"
app = marimo.App(width="full")


# ── Cell 1: Imports & paths ──────────────────────────────────────────
@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ROOT = Path(__file__).resolve().parents[1]
    METRO_DIR = ROOT / "data" / "raw" / "metropt3"

    STEEL = "#315c72"
    SLATE = "#8aa1af"
    AMBER = "#cf8f3d"
    ALERT = "#b45309"
    RED = "#b91c1c"
    BG = "#f5f7f9"

    return AMBER, ALERT, BG, METRO_DIR, RED, ROOT, STEEL, SLATE, go, make_subplots, mo, np, pd


# ── Cell 2: Data loading ─────────────────────────────────────────────
@app.cell
def _(METRO_DIR, np, pd):
    def _find_data_file(directory):
        if not directory.exists():
            return None
        for suffix in ("*.parquet", "*.feather", "*.csv"):
            candidates = sorted(directory.glob(suffix))
            if candidates:
                return candidates[0]
        return None

    def _load_table(path, target_rows=120_000):
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".feather":
            df = pd.read_feather(path)
        else:
            file_size_mb = path.stat().st_size / 1_000_000
            if file_size_mb < 30:
                df = pd.read_csv(path)
            else:
                chunks = []
                for chunk in pd.read_csv(path, chunksize=50_000):
                    step = max(1, len(chunk) // 350)
                    chunks.append(chunk.iloc[::step].copy())
                df = pd.concat(chunks, ignore_index=True)
        if len(df) > target_rows:
            step = max(1, len(df) // target_rows)
            df = df.iloc[::step].copy()
        return df

    def _generate_demo_data():
        """Synthetic compressor data with injected degradation events."""
        rng = np.random.default_rng(42)
        n = 24 * 120  # 120 days hourly
        ts = pd.date_range("2026-01-01", periods=n, freq="h")
        t = np.arange(n)
        df = pd.DataFrame({
            "timestamp": ts,
            "TP2": 8.0 + 0.6 * np.sin(t / 48) + rng.normal(0, 0.3, n),
            "TP3": 8.8 + 0.5 * np.sin(t / 36) + rng.normal(0, 0.25, n),
            "H1": 55 + 4 * np.sin(t / 72) + rng.normal(0, 1.5, n),
            "DV_pressure": 5.2 + 0.4 * np.sin(t / 24) + rng.normal(0, 0.15, n),
            "Oil_temperature": 64 + 2.5 * np.sin(t / 36) + rng.normal(0, 0.8, n),
            "Motor_current": 32 + 1.2 * np.sin(t / 24) + rng.normal(0, 0.4, n),
            "COMP": 1.0,
        })
        # Inject degradation events
        for start_day in (30, 65, 95):
            s = start_day * 24
            e = s + 48
            ramp = np.linspace(0, 1, e - s)
            df.loc[s:e - 1, "TP2"] += 1.5 * ramp + rng.normal(0, 0.5, e - s)
            df.loc[s:e - 1, "Oil_temperature"] += 4 * ramp
            df.loc[s:e - 1, "Motor_current"] += 3 * ramp
            df.loc[s:e - 1, "DV_pressure"] -= 1.5 * ramp
        return df

    source_path = _find_data_file(METRO_DIR)
    if source_path is not None:
        raw_df = _load_table(source_path)
        data_mode = "file"
        data_source = source_path.name
    else:
        raw_df = _generate_demo_data()
        data_mode = "demo"
        data_source = "Synthetic demo data"

    raw_df.columns = [c.strip() for c in raw_df.columns]

    # Detect timestamp
    _ts_candidates = ["timestamp", "datetime", "date_time", "time"]
    ts_col = next((c for c in raw_df.columns if c.lower() in _ts_candidates), None)
    if ts_col:
        raw_df[ts_col] = pd.to_datetime(raw_df[ts_col], errors="coerce")
        raw_df = raw_df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    else:
        ts_col = "timestamp"
        raw_df[ts_col] = pd.date_range("2026-01-01", periods=len(raw_df), freq="h")

    # Detect COMP column
    comp_col = next((c for c in raw_df.columns if c.upper() == "COMP"), None)

    # Identify continuous sensor columns (top 6 by variability)
    _skip = {ts_col, comp_col} if comp_col else {ts_col}
    numeric_cols = [
        c for c in raw_df.select_dtypes(include="number").columns
        if c not in _skip and not c.lower().startswith("unnamed") and "index" not in c.lower()
    ]
    continuous_cols = [c for c in numeric_cols if raw_df[c].nunique() > 2]
    sensor_cols = list(raw_df[continuous_cols].std().sort_values(ascending=False).index[:6])

    return comp_col, continuous_cols, data_mode, data_source, raw_df, sensor_cols, ts_col


# ── Cell 3: Header ───────────────────────────────────────────────────
@app.cell
def _(data_mode, data_source, mo, sensor_cols):
    _mode_badge = (
        '<span style="color:#b45309;font-weight:600">DEMO</span>'
        if data_mode == "demo"
        else '<span style="color:#315c72;font-weight:600">LIVE</span>'
    )
    mo.md(f"""
    # Signal Processing & State Estimation for Predictive Maintenance

    **Data**: {data_source} ({_mode_badge}) &mdash; **Sensors**: {', '.join(sensor_cols)}

    This notebook applies techniques from Shuai Guo's *"Data-Driven PdM In a Nutshell"*
    and Marcin Stasko's *"Understanding Predictive Maintenance"* series to improve
    fault detection and RUL estimation on compressor telemetry.

    **Pipeline**: Raw Signal &rarr; Denoise &rarr; Feature Engineering (time + spectral) &rarr; Stationarity Testing &rarr; State Estimation (Kalman / HMM)
    """)
    return


# ── Cell 4: Sensor picker ────────────────────────────────────────────
@app.cell
def _(mo, sensor_cols):
    sensor_picker = mo.ui.dropdown(
        options={col: col for col in sensor_cols},
        value=sensor_cols[0],
        label="Focus sensor",
    )
    window_picker = mo.ui.slider(
        start=30, stop=360, step=30, value=60,
        label="Window size (samples)",
    )
    mo.hstack([sensor_picker, window_picker], gap=1)
    return sensor_picker, window_picker


# ── Cell 5: Signal denoising ─────────────────────────────────────────
@app.cell
def _(comp_col, go, make_subplots, mo, np, pd, raw_df, sensor_picker, ts_col, STEEL, AMBER, SLATE):
    import pywt
    from scipy.signal import butter, filtfilt

    _col = sensor_picker.value
    _raw = raw_df[_col].astype(float).interpolate(limit_direction="both").values

    # --- Wavelet denoising (Daubechies-4, level 4, soft threshold) ---
    def wavelet_denoise(signal, wavelet="db4", level=4):
        signal = np.array(signal, dtype=float, copy=True)
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        # Universal threshold (VisuShrink)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        denoised_coeffs = [coeffs[0]]  # keep approximation
        for c in coeffs[1:]:
            denoised_coeffs.append(pywt.threshold(c, threshold, mode="soft"))
        return pywt.waverec(denoised_coeffs, wavelet)[: len(signal)]

    # --- Butterworth low-pass filter ---
    def butterworth_lowpass(signal, cutoff_ratio=0.05, order=4):
        b, a = butter(order, cutoff_ratio, btype="low")
        return filtfilt(b, a, signal)

    wavelet_clean = wavelet_denoise(_raw)
    butter_clean = butterworth_lowpass(_raw)

    # Store for downstream cells
    denoised_series = pd.Series(wavelet_clean, index=raw_df.index, name=_col + "_denoised")

    # --- Comparison plot ---
    _ts = raw_df[ts_col]
    # Downsample for plotting
    _n = len(_ts)
    _step = max(1, _n // 2000)
    _idx = np.arange(0, _n, _step)

    fig_denoise = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["Wavelet Denoising (db4, soft threshold)", "Butterworth Low-Pass Filter"],
        vertical_spacing=0.08,
    )
    for row, (clean, label) in enumerate(
        [(wavelet_clean, "Wavelet"), (butter_clean, "Butterworth")], 1
    ):
        fig_denoise.add_trace(
            go.Scatter(x=_ts.iloc[_idx], y=_raw[_idx], mode="lines",
                       name="Raw", line=dict(color=SLATE, width=0.5), opacity=0.5,
                       showlegend=(row == 1)),
            row=row, col=1,
        )
        fig_denoise.add_trace(
            go.Scatter(x=_ts.iloc[_idx], y=clean[_idx], mode="lines",
                       name=label, line=dict(color=STEEL if row == 1 else AMBER, width=1.5)),
            row=row, col=1,
        )
    fig_denoise.update_layout(
        height=500, template="plotly_white",
        title_text=f"Signal Denoising: {_col}",
        margin=dict(t=60, b=30),
    )

    # Noise reduction stats
    _raw_std = np.std(_raw)
    _wav_residual_std = np.std(_raw - wavelet_clean)
    _but_residual_std = np.std(_raw - butter_clean)

    denoise_stats = mo.md(f"""
    | Method | Noise Removed (std) | Signal Retained |
    |--------|-------------------|-----------------|
    | Wavelet (db4) | {_wav_residual_std:.4f} | {(1 - _wav_residual_std / _raw_std) * 100:.1f}% |
    | Butterworth LP | {_but_residual_std:.4f} | {(1 - _but_residual_std / _raw_std) * 100:.1f}% |
    """)

    mo.vstack([mo.ui.plotly(fig_denoise), denoise_stats])
    return butter_clean, butterworth_lowpass, denoised_series, wavelet_clean, wavelet_denoise


# ── Cell 6: Time-domain feature engineering ──────────────────────────
@app.cell
def _(denoised_series, go, mo, np, pd, raw_df, sensor_cols, ts_col, wavelet_denoise, window_picker, STEEL, AMBER, ALERT, RED, SLATE):
    from scipy.stats import kurtosis as scipy_kurtosis
    from scipy.stats import skew as scipy_skew

    _window = window_picker.value

    def compute_time_features(signal, window):
        """Compute rolling time-domain features for a 1-D signal."""
        s = pd.Series(signal)
        out = pd.DataFrame(index=s.index)
        out["rms"] = s.rolling(window, min_periods=window // 2).apply(
            lambda x: np.sqrt(np.mean(x**2)), raw=True
        )
        out["peak_to_peak"] = s.rolling(window, min_periods=window // 2).apply(
            lambda x: np.ptp(x), raw=True
        )
        out["crest_factor"] = s.rolling(window, min_periods=window // 2).apply(
            lambda x: np.max(np.abs(x)) / (np.sqrt(np.mean(x**2)) + 1e-12), raw=True
        )
        out["kurtosis"] = s.rolling(window, min_periods=window // 2).apply(
            lambda x: scipy_kurtosis(x, fisher=False), raw=True
        )
        out["skewness"] = s.rolling(window, min_periods=window // 2).apply(
            lambda x: scipy_skew(x), raw=True
        )
        return out

    # Compute for all sensors (using wavelet-denoised data)
    all_time_features = {}
    for col in sensor_cols:
        _raw_signal = raw_df[col].astype(float).interpolate(limit_direction="both").values
        _clean = wavelet_denoise(_raw_signal)
        all_time_features[col] = compute_time_features(_clean, _window)

    # Plot the focus sensor's features
    _focus = denoised_series.name.replace("_denoised", "")
    _feats = all_time_features[_focus]
    _ts = raw_df[ts_col]
    _step = max(1, len(_ts) // 2000)
    _idx = np.arange(0, len(_ts), _step)

    _feat_names = ["rms", "kurtosis", "crest_factor", "skewness", "peak_to_peak"]
    _colors = [STEEL, AMBER, ALERT, RED, SLATE]

    from plotly.subplots import make_subplots as _ms
    fig_time = _ms(rows=len(_feat_names), cols=1, shared_xaxes=True,
                   subplot_titles=[f.replace("_", " ").title() for f in _feat_names],
                   vertical_spacing=0.04)

    for i, (feat, color) in enumerate(zip(_feat_names, _colors), 1):
        _vals = _feats[feat].values
        fig_time.add_trace(
            go.Scatter(x=_ts.iloc[_idx], y=_vals[_idx], mode="lines",
                       name=feat.replace("_", " ").title(),
                       line=dict(color=color, width=1.2)),
            row=i, col=1,
        )
        # Add reference line for kurtosis (normal = 3)
        if feat == "kurtosis":
            fig_time.add_hline(y=3, line_dash="dash", line_color="gray",
                               annotation_text="Normal (3)", row=i, col=1)

    fig_time.update_layout(
        height=800, template="plotly_white", showlegend=False,
        title_text=f"Time-Domain Features: {_focus} (window={_window})",
        margin=dict(t=60, b=30),
    )

    mo.vstack([
        mo.md("## Time-Domain Feature Engineering"),
        mo.md(f"""
        Rolling window features computed on **wavelet-denoised** signal.
        - **RMS**: signal energy (imbalance indicator)
        - **Kurtosis**: impulsiveness (healthy ~ 3, faults > 3)
        - **Crest Factor**: peak/RMS ratio (spiky fault detection)
        - **Skewness**: directional asymmetry (rubbing vs bearing defects)
        - **Peak-to-Peak**: total displacement range
        """),
        mo.ui.plotly(fig_time),
    ])
    return all_time_features, compute_time_features


# ── Cell 7: Spectral feature engineering ─────────────────────────────
@app.cell
def _(denoised_series, go, mo, np, pd, raw_df, sensor_cols, ts_col, wavelet_denoise, window_picker, STEEL, AMBER, ALERT, SLATE):
    from scipy.signal import welch as scipy_welch

    _window = window_picker.value

    def compute_spectral_features(signal, window, fs=1.0):
        """Compute rolling spectral features for a 1-D signal."""
        s = pd.Series(signal)
        out = pd.DataFrame(index=s.index)

        def _spectral_stats(x):
            freqs, psd = scipy_welch(x, fs=fs, nperseg=min(len(x), 256))
            psd = psd + 1e-12  # avoid log(0)
            # Dominant frequency
            dominant_freq = freqs[np.argmax(psd)]
            # Spectral entropy (Shannon entropy of normalized PSD)
            psd_norm = psd / psd.sum()
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
            # Spectral kurtosis (kurtosis of PSD distribution)
            mean_psd = np.mean(psd)
            std_psd = np.std(psd) + 1e-12
            spectral_kurt = np.mean(((psd - mean_psd) / std_psd) ** 4)
            # Band power ratio (low vs high frequency)
            mid = len(freqs) // 2
            low_power = np.sum(psd[:mid])
            high_power = np.sum(psd[mid:]) + 1e-12
            band_ratio = low_power / high_power
            return pd.Series({
                "dominant_freq": dominant_freq,
                "spectral_entropy": spectral_entropy,
                "spectral_kurtosis": spectral_kurt,
                "band_power_ratio": band_ratio,
            })

        # Compute in chunks for efficiency
        rows = []
        for start in range(0, len(s), max(1, window // 4)):
            end = min(start + window, len(s))
            chunk = s.iloc[start:end].values
            center = start + len(chunk) // 2
            if len(chunk) < window // 2:
                rows.append((center, np.nan, np.nan, np.nan, np.nan))
            else:
                stats = _spectral_stats(chunk)
                rows.append((center, stats["dominant_freq"], stats["spectral_entropy"],
                             stats["spectral_kurtosis"], stats["band_power_ratio"]))

        spec_df = pd.DataFrame(rows, columns=["_center", "dominant_freq", "spectral_entropy",
                                               "spectral_kurtosis", "band_power_ratio"])
        # Reindex to original index via interpolation
        out = pd.DataFrame(index=s.index, dtype=float)
        for feat in ["dominant_freq", "spectral_entropy", "spectral_kurtosis", "band_power_ratio"]:
            interp = np.interp(
                np.arange(len(s)),
                spec_df["_center"].values,
                spec_df[feat].values,
            )
            out[feat] = interp
        return out

    # Compute for all sensors
    all_spectral_features = {}
    for col in sensor_cols:
        _raw_signal = raw_df[col].astype(float).interpolate(limit_direction="both").values
        _clean = wavelet_denoise(_raw_signal)
        all_spectral_features[col] = compute_spectral_features(_clean, _window)

    # Plot focus sensor
    _focus = denoised_series.name.replace("_denoised", "")
    _spec = all_spectral_features[_focus]
    _ts = raw_df[ts_col]
    _step = max(1, len(_ts) // 2000)
    _idx = np.arange(0, len(_ts), _step)

    _feat_names = ["dominant_freq", "spectral_entropy", "spectral_kurtosis", "band_power_ratio"]
    _labels = ["Dominant Frequency", "Spectral Entropy", "Spectral Kurtosis", "Band Power Ratio (Low/High)"]
    _colors = [STEEL, AMBER, ALERT, SLATE]

    from plotly.subplots import make_subplots as _ms2
    fig_spec = _ms2(rows=4, cols=1, shared_xaxes=True,
                    subplot_titles=_labels, vertical_spacing=0.05)

    for i, (feat, label, color) in enumerate(zip(_feat_names, _labels, _colors), 1):
        _vals = _spec[feat].values
        fig_spec.add_trace(
            go.Scatter(x=_ts.iloc[_idx], y=_vals[_idx], mode="lines",
                       name=label, line=dict(color=color, width=1.2)),
            row=i, col=1,
        )

    fig_spec.update_layout(
        height=700, template="plotly_white", showlegend=False,
        title_text=f"Spectral Features: {_focus} (window={_window})",
        margin=dict(t=60, b=30),
    )

    # PSD snapshot of the full signal
    _full_signal = wavelet_denoise(
        raw_df[_focus].astype(float).interpolate(limit_direction="both").values
    )
    _freqs, _psd = scipy_welch(_full_signal, fs=1.0, nperseg=min(len(_full_signal), 1024))

    fig_psd = go.Figure()
    fig_psd.add_trace(go.Scatter(x=_freqs, y=_psd, mode="lines",
                                 line=dict(color=STEEL, width=1.5),
                                 name="PSD"))
    fig_psd.update_layout(
        height=300, template="plotly_white",
        title_text=f"Power Spectral Density: {_focus}",
        xaxis_title="Frequency", yaxis_title="Power",
        margin=dict(t=50, b=40),
    )

    mo.vstack([
        mo.md("## Spectral Feature Engineering"),
        mo.md("""
        Frequency-domain features computed via Welch's method on sliding windows:
        - **Dominant Frequency**: characteristic operating/fault frequency
        - **Spectral Entropy**: disorder in spectrum (low = periodic fault, high = broadband noise)
        - **Spectral Kurtosis**: impulsiveness in frequency domain
        - **Band Power Ratio**: low-freq vs high-freq energy distribution
        """),
        mo.ui.plotly(fig_spec),
        mo.md("### Full-Signal Power Spectral Density"),
        mo.ui.plotly(fig_psd),
    ])
    return all_spectral_features, compute_spectral_features


# ── Cell 8: Stationarity testing ─────────────────────────────────────
@app.cell
def _(mo, np, pd, raw_df, sensor_cols, wavelet_denoise):
    from statsmodels.tsa.stattools import adfuller, kpss

    stationarity_results = []
    for col in sensor_cols:
        _signal = wavelet_denoise(
            raw_df[col].astype(float).interpolate(limit_direction="both").values
        )
        # Subsample for speed (ADF/KPSS are O(n^2))
        _sub = _signal[::max(1, len(_signal) // 5000)]

        # ADF test (H0: unit root / non-stationary)
        try:
            adf_stat, adf_p, *_ = adfuller(_sub, maxlag=20, autolag="AIC")
        except Exception:
            adf_stat, adf_p = np.nan, np.nan

        # KPSS test (H0: stationary)
        try:
            kpss_stat, kpss_p, *_ = kpss(_sub, regression="c", nlags="auto")
        except Exception:
            kpss_stat, kpss_p = np.nan, np.nan

        adf_stationary = adf_p < 0.05 if not np.isnan(adf_p) else None
        kpss_stationary = kpss_p > 0.05 if not np.isnan(kpss_p) else None

        if adf_stationary and kpss_stationary:
            verdict = "Stationary"
        elif not adf_stationary and not kpss_stationary:
            verdict = "Non-stationary"
        elif adf_stationary and not kpss_stationary:
            verdict = "Trend-stationary"
        else:
            verdict = "Difference-stationary"

        stationarity_results.append({
            "Sensor": col,
            "ADF Statistic": round(adf_stat, 3) if not np.isnan(adf_stat) else "N/A",
            "ADF p-value": round(adf_p, 4) if not np.isnan(adf_p) else "N/A",
            "ADF Stationary": adf_stationary,
            "KPSS Statistic": round(kpss_stat, 3) if not np.isnan(kpss_stat) else "N/A",
            "KPSS p-value": round(kpss_p, 4) if not np.isnan(kpss_p) else "N/A",
            "KPSS Stationary": kpss_stationary,
            "Verdict": verdict,
        })

    stationarity_df = pd.DataFrame(stationarity_results)

    mo.vstack([
        mo.md("## Stationarity Testing"),
        mo.md("""
        Tests whether each sensor's signal is stationary (stable statistics over time)
        or non-stationary (drifting — e.g., degradation trend).

        - **ADF** (Augmented Dickey-Fuller): H0 = non-stationary. Low p-value = stationary.
        - **KPSS**: H0 = stationary. High p-value = stationary.
        - **Non-stationary** sensors have real degradation trends worth modeling.
        - **Stationary** sensors are stable — linear trend extrapolation is unreliable for these.
        """),
        stationarity_df,
    ])
    return stationarity_df, stationarity_results


# ── Cell 9: Kalman filter state estimation ───────────────────────────
@app.cell
def _(all_time_features, go, mo, np, pd, raw_df, sensor_cols, ts_col, STEEL, AMBER, RED, SLATE):
    from filterpy.kalman import KalmanFilter

    # Build a health index from time-domain features across all sensors
    # Combine kurtosis + crest factor as proxy for overall degradation
    _n = len(raw_df)
    health_signals = pd.DataFrame(index=raw_df.index)

    for col in sensor_cols:
        feats = all_time_features[col]
        # Normalize each feature to 0-1 range
        for feat in ["kurtosis", "crest_factor", "rms"]:
            series = feats[feat].bfill().ffill()
            _min, _max = series.min(), series.max()
            if _max - _min > 1e-8:
                health_signals[f"{col}_{feat}"] = (series - _min) / (_max - _min)
            else:
                health_signals[f"{col}_{feat}"] = 0.0

    # Composite observation: mean of all normalized features
    observation = health_signals.mean(axis=1).values

    # --- Kalman Filter ---
    # State: [health_level, health_trend]
    # Observation: composite feature index
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[observation[0]], [0.0]])  # initial state
    kf.F = np.array([[1, 1], [0, 1]])  # state transition (linear trend)
    kf.H = np.array([[1, 0]])  # observation matrix
    kf.P *= 10  # initial uncertainty
    kf.R = np.array([[0.1]])  # measurement noise
    kf.Q = np.array([[0.001, 0], [0, 0.0001]])  # process noise

    kalman_states = np.zeros((_n, 2))
    kalman_covs = np.zeros(_n)
    for i in range(_n):
        kf.predict()
        kf.update(np.array([[observation[i]]]))
        kalman_states[i] = kf.x.flatten()
        kalman_covs[i] = kf.P[0, 0]

    health_level = kalman_states[:, 0]
    health_trend = kalman_states[:, 1]
    uncertainty = 2 * np.sqrt(kalman_covs)  # 95% confidence

    # --- Plot ---
    _ts = raw_df[ts_col]
    _step = max(1, _n // 2000)
    _idx = np.arange(0, _n, _step)

    from plotly.subplots import make_subplots as _ms3
    fig_kalman = _ms3(rows=2, cols=1, shared_xaxes=True,
                      subplot_titles=["Health State (Kalman Filtered)", "Health Trend (Rate of Change)"],
                      vertical_spacing=0.1)

    # Health level with uncertainty band
    fig_kalman.add_trace(
        go.Scatter(x=_ts.iloc[_idx], y=(health_level + uncertainty)[_idx],
                   mode="lines", line=dict(width=0), showlegend=False),
        row=1, col=1,
    )
    fig_kalman.add_trace(
        go.Scatter(x=_ts.iloc[_idx], y=(health_level - uncertainty)[_idx],
                   mode="lines", line=dict(width=0), fill="tonexty",
                   fillcolor="rgba(49,92,114,0.15)", name="95% Confidence"),
        row=1, col=1,
    )
    fig_kalman.add_trace(
        go.Scatter(x=_ts.iloc[_idx], y=observation[_idx], mode="lines",
                   name="Raw Observation", line=dict(color=SLATE, width=0.5), opacity=0.4),
        row=1, col=1,
    )
    fig_kalman.add_trace(
        go.Scatter(x=_ts.iloc[_idx], y=health_level[_idx], mode="lines",
                   name="Kalman State", line=dict(color=STEEL, width=2)),
        row=1, col=1,
    )

    # Health trend
    fig_kalman.add_trace(
        go.Scatter(x=_ts.iloc[_idx], y=health_trend[_idx], mode="lines",
                   name="Health Trend", line=dict(color=AMBER, width=1.5)),
        row=2, col=1,
    )
    fig_kalman.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    fig_kalman.update_layout(
        height=550, template="plotly_white",
        title_text="Kalman Filter: Hidden Health State Estimation",
        margin=dict(t=60, b=30),
    )

    mo.vstack([
        mo.md("## State Estimation: Kalman Filter"),
        mo.md("""
        The Kalman filter estimates a **hidden health state** from noisy sensor observations.
        Our sensors (pressure, temperature, current) are *indirect observables* of compressor health
        (per Guo's classification). The filter tracks:

        - **Health Level**: smoothed estimate of overall equipment condition (0 = baseline, rising = degrading)
        - **Health Trend**: rate of change in health state (positive = worsening)
        - **Uncertainty Band**: 95% confidence interval on the health estimate

        Rising health level with positive trend = active degradation requiring attention.
        """),
        mo.ui.plotly(fig_kalman),
    ])
    return health_level, health_signals, health_trend, kalman_states, observation, uncertainty


# ── Cell 10: HMM health state classification ─────────────────────────
@app.cell
def _(go, health_signals, mo, np, pd, raw_df, ts_col, STEEL, AMBER, RED, SLATE, BG):
    from hmmlearn.hmm import GaussianHMM

    # Fit a 3-state HMM: Healthy, Degrading, Critical
    _features = health_signals.dropna()
    _feature_matrix = _features.values

    hmm = GaussianHMM(
        n_components=3, covariance_type="full",
        n_iter=100, random_state=42, tol=0.01,
    )
    hmm.fit(_feature_matrix)
    hidden_states = hmm.predict(_feature_matrix)

    # Map states to semantic labels by sorting on mean feature value
    state_means = [_feature_matrix[hidden_states == s].mean() for s in range(3)]
    state_order = np.argsort(state_means)  # lowest mean → healthiest
    state_map = {state_order[0]: "Healthy", state_order[1]: "Degrading", state_order[2]: "Critical"}
    state_colors = {"Healthy": STEEL, "Degrading": AMBER, "Critical": RED}

    labeled_states = pd.Series(
        [state_map[s] for s in hidden_states],
        index=_features.index,
        name="health_state",
    )

    # Reindex to full dataframe
    labeled_states_full = labeled_states.reindex(raw_df.index, fill_value="Unknown")

    # Plot
    _ts = raw_df[ts_col]
    _step = max(1, len(_ts) // 3000)
    _idx = np.arange(0, len(_ts), _step)

    fig_hmm = go.Figure()
    for state_name, color in state_colors.items():
        _mask = labeled_states_full.iloc[_idx] == state_name
        if _mask.any():
            _subset_idx = _idx[_mask.values]
            fig_hmm.add_trace(go.Scatter(
                x=_ts.iloc[_subset_idx],
                y=np.ones(len(_subset_idx)) if state_name == "Healthy" else
                  np.ones(len(_subset_idx)) * 2 if state_name == "Degrading" else
                  np.ones(len(_subset_idx)) * 3,
                mode="markers", marker=dict(color=color, size=3),
                name=state_name,
            ))

    fig_hmm.update_layout(
        height=250, template="plotly_white",
        title_text="Hidden Markov Model: Health State Classification",
        yaxis=dict(tickvals=[1, 2, 3], ticktext=["Healthy", "Degrading", "Critical"],
                   range=[0.5, 3.5]),
        margin=dict(t=50, b=30),
    )

    # Transition matrix
    trans = pd.DataFrame(
        hmm.transmat_,
        index=[state_map[state_order[i]] for i in range(3)],
        columns=[state_map[state_order[i]] for i in range(3)],
    ).round(3)

    # State distribution
    state_counts = labeled_states.value_counts()
    state_pct = (state_counts / len(labeled_states) * 100).round(1)

    mo.vstack([
        mo.md("## State Estimation: Hidden Markov Model"),
        mo.md("""
        The HMM classifies each time step into one of three discrete health states
        (**Healthy**, **Degrading**, **Critical**) based on the multi-sensor feature matrix.
        Unlike threshold-based rules, the HMM learns state boundaries and transition probabilities
        from the data itself.
        """),
        mo.ui.plotly(fig_hmm),
        mo.md("### State Transition Probabilities"),
        trans,
        mo.md("### Time Spent in Each State"),
        mo.md(f"""
        | State | % of Time |
        |-------|-----------|
        | Healthy | {state_pct.get('Healthy', 0):.1f}% |
        | Degrading | {state_pct.get('Degrading', 0):.1f}% |
        | Critical | {state_pct.get('Critical', 0):.1f}% |
        """),
    ])
    return hidden_states, hmm, labeled_states_full, state_map, trans


# ── Cell 11: Combined dashboard view ─────────────────────────────────
@app.cell
def _(go, health_level, labeled_states_full, mo, np, raw_df, sensor_picker, ts_col, uncertainty, wavelet_clean, STEEL, AMBER, RED, SLATE):
    _col = sensor_picker.value
    _ts = raw_df[ts_col]
    _n = len(_ts)
    _step = max(1, _n // 2000)
    _idx = np.arange(0, _n, _step)

    _raw_signal = raw_df[_col].astype(float).interpolate(limit_direction="both").values

    from plotly.subplots import make_subplots as _ms4
    fig_combined = _ms4(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=[
            f"Raw vs Denoised: {_col}",
            "Kalman Health State (with uncertainty)",
            "HMM Classification",
        ],
        vertical_spacing=0.07,
        row_heights=[0.35, 0.35, 0.15],
    )

    # Row 1: raw vs denoised
    fig_combined.add_trace(
        go.Scatter(x=_ts.iloc[_idx], y=_raw_signal[_idx], mode="lines",
                   name="Raw", line=dict(color=SLATE, width=0.5), opacity=0.4),
        row=1, col=1,
    )
    fig_combined.add_trace(
        go.Scatter(x=_ts.iloc[_idx], y=wavelet_clean[_idx], mode="lines",
                   name="Denoised", line=dict(color=STEEL, width=1.5)),
        row=1, col=1,
    )

    # Row 2: Kalman health state
    fig_combined.add_trace(
        go.Scatter(x=_ts.iloc[_idx], y=(health_level + uncertainty)[_idx],
                   mode="lines", line=dict(width=0), showlegend=False),
        row=2, col=1,
    )
    fig_combined.add_trace(
        go.Scatter(x=_ts.iloc[_idx], y=(health_level - uncertainty)[_idx],
                   mode="lines", line=dict(width=0), fill="tonexty",
                   fillcolor="rgba(49,92,114,0.15)", name="95% CI"),
        row=2, col=1,
    )
    fig_combined.add_trace(
        go.Scatter(x=_ts.iloc[_idx], y=health_level[_idx], mode="lines",
                   name="Health Level", line=dict(color=STEEL, width=2)),
        row=2, col=1,
    )

    # Row 3: HMM states as colored markers
    _state_colors = {"Healthy": STEEL, "Degrading": AMBER, "Critical": RED}
    _state_y = {"Healthy": 1, "Degrading": 2, "Critical": 3}
    for state_name, color in _state_colors.items():
        _mask = labeled_states_full.iloc[_idx] == state_name
        if _mask.any():
            _sub = _idx[_mask.values]
            fig_combined.add_trace(go.Scatter(
                x=_ts.iloc[_sub], y=[_state_y[state_name]] * len(_sub),
                mode="markers", marker=dict(color=color, size=3), name=state_name,
            ), row=3, col=1)

    fig_combined.update_layout(
        height=700, template="plotly_white",
        title_text="Combined Pipeline View",
        margin=dict(t=60, b=30),
    )
    fig_combined.update_yaxes(
        tickvals=[1, 2, 3], ticktext=["Healthy", "Degrading", "Critical"],
        range=[0.5, 3.5], row=3, col=1,
    )

    mo.vstack([
        mo.md("## Combined Pipeline View"),
        mo.md("""
        End-to-end: **Raw Signal** &rarr; **Wavelet Denoising** &rarr; **Feature Extraction** &rarr;
        **Kalman Health Estimate** + **HMM State Classification**
        """),
        mo.ui.plotly(fig_combined),
    ])
    return


if __name__ == "__main__":
    app.run()
