"""Fan benchmark using labeled source development clips and held-out target clips."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import stft
import soundfile as sf
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.benchmark import ROOT, report
from src.fan import recordings

CACHE = ROOT / "results/fan_supervised/spectral_features.npz"


def extract(source, display=False):
    audio, rate = sf.read(source, dtype="float32")
    if rate != 16000 or audio.shape != (160000,) or not np.isfinite(audio).all():
        raise ValueError("Expected a finite, 10-second, mono 16 kHz WAV.")
    hz, times, transform = stft(audio, rate, nperseg=4096, noverlap=2048, boundary=None)
    power = np.abs(transform) ** 2
    log = 10 * np.log10(power + 1e-12)
    profile = 10 * np.log10(power.mean(axis=1) + 1e-12)
    narrow = profile - gaussian_filter1d(profile, 6)
    centered = log - np.median(log, axis=1, keepdims=True)
    bands = np.unique(np.geomspace(1, len(hz), 80).astype(int))[:-1]
    band_power = np.add.reduceat(power, bands, axis=0)
    band = 10 * np.log10(band_power + 1e-12)
    shape = band - band.mean(axis=0, keepdims=True)
    f = np.r_[profile[:768:2], narrow[:768:2], gaussian_filter1d(profile, 5)[::12],
              band.mean(1), shape.mean(1), band.std(1),
              np.quantile(band, .9, axis=1), np.quantile(centered[:768:4], .9, axis=1)]
    return (f, audio, times, band) if display else f


def scores(model, x, domain="source"):
    if domain == "target" and "target_offset" in model:
        x = x - model["target_offset"]
    return model["estimator"].decision_function(x) if model["method"].endswith("svm") else (
        model["estimator"].predict_proba(x)[:, 1]
    )


def predict(source, model, domain="source"):
    features, audio, times, spectral = extract(source, display=True)
    value = float(scores(model, features[None], domain)[0])
    return value, value > model["threshold"], audio, times, spectral


def train(root):
    items = list(recordings(root))
    meta = pd.DataFrame([(p.name, sec, domain, split, label)
                         for p, sec, domain, split, label in items],
                        columns=["file", "section", "domain", "split", "actual"])
    signatures = np.array([f"{p.resolve()}|{p.stat().st_size}|{p.stat().st_mtime_ns}"
                           for p, *_ in items])
    with np.load(CACHE) if CACHE.exists() else _empty_cache() as saved:
        x = saved["x"] if "x" in saved and np.array_equal(saved["keys"], signatures) else None
    if x is None:
        with ThreadPoolExecutor(max_workers=4) as pool:
            features = []
            for index, vector in enumerate(pool.map(extract, (p for p, *_ in items)), 1):
                features.append(vector)
                if index % 500 == 0:
                    print(f"Prepared {index} fan clips", flush=True)
        x = np.stack(features)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        temporary = CACHE.with_name("spectral_features.tmp.npz")
        np.savez_compressed(temporary, keys=signatures, x=x)
        temporary.replace(CACHE)
    predictions, models = [], {}
    for section, group in meta.groupby("section"):
        source = group[(group.domain == "source") & (group.split == "test")]
        adapt, holdout = train_test_split(source.index, test_size=.30,
                                          stratify=source.actual, random_state=218)
        normal = group[(group.split == "train") & (group.domain == "source")].sample(
            n=250, random_state=218).index
        target_normal = group[(group.split == "train") & (group.domain == "target")].index
        offset = .75 * (np.median(x[target_normal], axis=0) - np.median(x[normal], axis=0))

        fit = np.r_[normal, adapt]
        test = group.loc[group.index.isin(holdout) |
                         ((group.domain == "target") & (group.split == "test"))]
        for method, estimator, threshold in (
            ("svm", make_pipeline(StandardScaler(), SVC(C=3, class_weight="balanced")), 0.),
            ("extra_trees", ExtraTreesClassifier(n_estimators=350, min_samples_leaf=2,
                                                  class_weight="balanced", random_state=218,
                                                  n_jobs=3), .5),
        ):
            estimator.fit(x[fit], meta.actual.iloc[fit])
            model = dict(method=method, estimator=estimator, threshold=threshold)
            models[section, method] = model
            predictions.append(test.assign(model=method, score=scores(model, x[test.index]),
                                           threshold=threshold))
            if method == "extra_trees":
                aligned = dict(model, method="aligned_extra_trees", target_offset=offset)
                models[section, aligned["method"]] = aligned
                for domain, domain_test in test.groupby("domain"):
                    predictions.append(domain_test.assign(
                        model=aligned["method"], score=scores(aligned, x[domain_test.index], domain),
                        threshold=threshold))

    prediction = pd.concat(predictions)
    report("fan_supervised", prediction, ["model", "section", "domain"], models)
    baseline = pd.read_csv(ROOT / "results/fan/predictions.csv", dtype={"section": str})
    from src.benchmark import classification_metrics
    baseline = baseline[baseline.file.isin(prediction.file)]
    comparisons = [dict(model=method, section=section, domain=domain, **classification_metrics(rows))
                   for (method, section, domain), rows in baseline.groupby(["model", "section", "domain"])]
    pd.DataFrame(comparisons).to_csv(CACHE.parent / "matched_baseline.csv", index=False)


def _empty_cache():
    from contextlib import nullcontext
    return nullcontext({})


if __name__ == "__main__":
    train(ROOT / "dataset/fan")
