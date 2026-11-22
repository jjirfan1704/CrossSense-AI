"""Bearing degradation scores calibrated on early chronological snapshots."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmark import ROOT, report

METHODS = ("rms", "relative_rms", "relative_peak", "impact_rms", "high_band", "high_band_ratio",
           "low_band", "mid_band", "envelope_rms")
FEATURES = ["Channel RMS", "RMS relative to reference", "Spectral peak relative to reference",
            "Upper-tail amplitude", "High-band RMS", "High-band RMS relative to reference",
            "Low-band RMS", "Mid-band RMS", "Band envelope modulation"]



def snapshot(source, channel, rate):
    data = np.loadtxt(source, ndmin=2)
    if data.shape[1] not in (4, 8) or not 1 <= channel <= data.shape[1] or len(data) < 256:
        raise ValueError("Expected an IMS snapshot with the configured channel.")
    if not np.isfinite(data).all():
        raise ValueError("Vibration samples must be finite.")
    signal = data[:, channel - 1]
    rms = np.sqrt(np.mean(data ** 2, axis=0)).clip(1e-10)
    others = np.arange(data.shape[1]) != channel - 1
    center = data - data.mean(axis=0)
    transformed = np.fft.rfft(center * np.hanning(len(data))[:, None], axis=0)
    spectrum = np.abs(transformed) ** 2
    hz = np.fft.rfftfreq(len(data), 1 / rate)
    def band_power(low, high):
        return np.sqrt(spectrum[(hz >= low) & (hz < high)].sum(axis=0)).clip(1e-10) / len(data)
    high = band_power(4000, 9000)
    carrier = transformed[:, channel - 1].copy()
    carrier[(hz < 2000) | (hz > 8000)] = 0
    transient = np.fft.irfft(carrier, n=len(data))
    blocks = np.array_split(transient ** 2, 64)
    envelope = np.std([np.sqrt(block.mean()) for block in blocks])

    peak = np.max(spectrum[1:], axis=0).clip(1e-10)
    return np.array([rms[channel - 1], rms[channel - 1] / np.median(rms[others]),
                     peak[channel - 1] / np.median(peak[others]),
                     np.quantile(np.abs(center[:, channel - 1]), .995),
                     high[channel - 1], high[channel - 1] / np.median(high[others]),
                     band_power(100, 1000)[channel - 1],
                     band_power(1000, 4000)[channel - 1], envelope]), signal


def score(model, values):
    return values[:, model["feature"]]


def predict(source, model):
    features, signal = snapshot(source, model["channel"], model["rate"])
    value = float(score(model, features[None])[0])
    return value, value > model["threshold"], dict(zip(FEATURES, features)), signal


def train(root, channel=1, rate=20000):
    rows = []
    for path in Path(root).iterdir():
        try:
            rows.append((datetime.strptime(path.name, "%Y.%m.%d.%H.%M.%S"), path))
        except ValueError:
            continue
    rows.sort()
    x = np.stack([snapshot(path, channel, rate)[0] for _, path in rows])
    early, start, late = [int(len(x) * f) for f in (.2, .3, .9)]
    test = pd.DataFrame(dict(timestamp=[t for t, _ in rows[start:]],
                             file=[p.name for _, p in rows[start:]], rms=x[start:, 0]))
    test["actual"] = np.nan
    test.loc[late - start:, "actual"] = 1
    results, models = [], {}
    for feature, method in enumerate(METHODS):
        threshold = float(np.quantile(x[early:start, feature], .99))
        model = dict(feature=feature, channel=channel, rate=rate, threshold=threshold,
                     run=str(root.resolve()))
        models[method] = model
        row = test.assign(model=method, score=score(model, x[start:]), threshold=threshold)
        row["persistent_alarm"] = (row.score > row.threshold).rolling(3).sum().eq(3)
        results.append(row)
    report("bearing_reference", pd.concat(results), ["model"], models)


if __name__ == "__main__":
    train(ROOT / "dataset/bearing/IMS/2nd_test")
