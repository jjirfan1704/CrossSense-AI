"""Fan development benchmark and WAV inference."""
import hashlib
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import stft
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.benchmark import ROOT, report

NAME = re.compile(r"section_(\d+)_((?:source|target))_(train|test)_(normal|anomaly)_\d+(?:_.*)?\.wav")


def recordings(root):
    for path in sorted(Path(root).rglob("*.wav")):
        match = NAME.fullmatch(path.name)
        if not match:
            raise ValueError(f"Unexpected fan filename: {path.name}")
        section, domain, split, label = match.groups()
        folder = "train" if split == "train" else f"{domain}_test"
        if path.parent.name != folder or (split == "train" and label != "normal"):
            raise ValueError(f"Invalid fan split or label: {path}")
        yield path, section, domain, split, int(label == "anomaly")


def audio_features(source):
    audio, rate = sf.read(source, dtype="float32")
    if rate != 16000 or audio.shape != (160000,) or not np.isfinite(audio).all():
        raise ValueError("Expected a finite, 10-second, mono 16 kHz WAV.")
    frequencies, times, transform = stft(audio, rate, nperseg=1024, noverlap=512)
    power = np.abs(transform) ** 2
    bands = np.unique(np.geomspace(1, len(frequencies), 65).astype(int))[:-1]
    spectral = 10 * np.log10(np.add.reduceat(power, bands, axis=0) + 1e-12)
    features = np.r_[spectral.mean(axis=1), spectral.std(axis=1),
                     np.quantile(spectral, .90, axis=1)]
    return features, audio, frequencies, times, spectral


def score(model, x):
    scaled = model["scaler"].transform(x)
    estimator = model["estimator"]
    if model["method"] == "neighbors":
        return estimator.kneighbors(scaled, return_distance=True)[0].mean(axis=1)
    return estimator.mahalanobis(scaled)


def train(root):
    rows, features, hashes = [], [], {}
    for path, section, domain, split, label in recordings(root):
        digest = hashlib.sha256(path.read_bytes()).digest()
        previous = hashes.setdefault(digest, split)
        if previous != split:
            raise ValueError(f"Byte-identical recording crosses train/test: {path}")
        rows.append((path.name, section, domain, split, label))
        features.append(audio_features(path)[0])
        if len(rows) % 500 == 0:
            print(f"Extracted {len(rows)} fan recordings", flush=True)
    if not rows:
        raise ValueError(f"No fan WAVs under {root}")
    meta = pd.DataFrame(rows, columns=["file", "section", "domain", "split", "actual"])
    x = np.stack(features)
    models, predictions = {}, []
    for section, group in meta.groupby("section"):
        source_train = group[(group.split == "train") & (group.domain == "source")]
        fit, calibration = train_test_split(source_train.index, test_size=.2, random_state=42)
        fit = np.r_[fit, group[(group.split == "train") & (group.domain == "target")].index]
        test = group[group.split == "test"]
        scaler = StandardScaler().fit(x[fit])
        for method, estimator in (("neighbors", NearestNeighbors(n_neighbors=10)),
                                  ("covariance", LedoitWolf())):
            estimator.fit(scaler.transform(x[fit]))
            model = dict(method=method, estimator=estimator, scaler=scaler)
            model["threshold"] = float(np.quantile(score(model, x[calibration]), .95))
            models[section, method] = model
            result = test.copy()
            result["model"] = method
            result["score"] = score(model, x[test.index])
            result["threshold"] = model["threshold"]
            predictions.append(result)
    report("fan", pd.concat(predictions), ["model", "section", "domain"], models)


def predict(source, model):
    features, audio, frequencies, times, spectral = audio_features(source)
    value = float(score(model, features.reshape(1, -1))[0])
    return value, value > model["threshold"], audio, times, spectral
