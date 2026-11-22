"""Chronological condition monitoring for a NASA IMS bearing channel."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from src.benchmark import report

FEATURES = ["rms", "std", "variance", "skewness", "kurtosis", "peak_to_peak",
            "mean_square", "crest_factor", "spectral_centroid_hz", "high_band_fraction"]


def snapshot(source, channel, rate):
    data = np.loadtxt(source, ndmin=2)
    if data.shape[1] not in (4, 8) or data.shape[0] < 256:
        raise ValueError("Expected a NASA IMS snapshot with four or eight vibration channels.")
    if not 1 <= channel <= data.shape[1] or not np.isfinite(data).all() or rate <= 0:
        raise ValueError("Invalid channel, vibration values, or sample rate.")
    x = data[:, channel - 1]
    rms = np.sqrt(np.mean(x ** 2))
    centered = x - x.mean()
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(len(x)))) ** 2
    hz = np.fft.rfftfreq(len(x), 1 / rate)
    total = max(spectrum.sum(), np.finfo(float).tiny)
    features = [rms, x.std(), x.var(), skew(x) if x.std() else 0,
                kurtosis(x) if x.std() else 0, np.ptp(x), rms ** 2,
                np.max(np.abs(x)) / rms if rms else 0,
                (spectrum @ hz) / total, spectrum[hz >= rate / 4].sum() / total]
    return np.array(features), x


def score(model, x):
    x = model["scaler"].transform(x)
    estimator = model["estimator"]
    if model["method"] == "isolation":
        return -estimator.score_samples(x)
    return estimator.mahalanobis(x)


def train(root, channel, rate):
    rows = []
    for path in Path(root).iterdir():
        if not path.is_file():
            continue
        try:
            timestamp = datetime.strptime(path.name, "%Y.%m.%d.%H.%M.%S")
        except ValueError:
            continue
        rows.append((timestamp, path))
    rows.sort()
    if len(rows) < 50:
        raise ValueError("Choose a single IMS run folder containing at least 50 timestamped snapshots.")
    x = np.stack([snapshot(path, channel, rate)[0] for _, path in rows])
    fit_end, calibrate_end, late = [int(len(x) * fraction) for fraction in (.2, .3, .9)]
    scaler = RobustScaler().fit(x[:fit_end])
    models, predictions = {}, []
    test = pd.DataFrame({"timestamp": [t for t, _ in rows[calibrate_end:]],
                         "file": [p.name for _, p in rows[calibrate_end:]],
                         "rms": x[calibrate_end:, 0]})
    test["actual"] = np.nan
    test.loc[late - calibrate_end:, "actual"] = 1
    for method, estimator in (("covariance", LedoitWolf()),
                              ("isolation", IsolationForest(random_state=42))):
        estimator.fit(scaler.transform(x[:fit_end]))
        model = dict(method=method, scaler=scaler, estimator=estimator, rate=rate,
                     channel=channel, run=str(Path(root).resolve()))
        validation = score(model, x[fit_end:calibrate_end])
        model["threshold"] = float(np.quantile(validation, .95))
        models[method] = model
        scores = score(model, x[calibrate_end:])
        result = test.assign(model=method, score=scores, threshold=model["threshold"])
        result["persistent_alarm"] = (result.score > result.threshold).rolling(3).sum().eq(3)
        predictions.append(result)
    report("bearing", pd.concat(predictions), ["model"], models)


def predict(source, model):
    features, signal = snapshot(source, model["channel"], model["rate"])
    value = float(score(model, features.reshape(1, -1))[0])
    return value, value > model["threshold"], dict(zip(FEATURES, features)), signal
