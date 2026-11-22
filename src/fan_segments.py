"""Segment-level fan classifiers with clip-level, held-out evaluation."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.benchmark import ROOT, report
from src.fan import recordings
from src.fan_supervised import extract as display_features


def features(source):
    audio, rate = sf.read(source, dtype="float32")
    if rate != 16000 or audio.shape != (160000,) or not np.isfinite(audio).all():
        raise ValueError("Expected a finite ten-second mono 16 kHz WAV.")
    clips = audio.reshape(5, -1)
    _, power = welch(clips, rate, nperseg=4096)
    log = 10 * np.log10(power + 1e-12)
    smooth = gaussian_filter1d(log, 7, axis=1)
    bands = np.unique(np.geomspace(1, power.shape[1], 74).astype(int))[:-1]
    energies = 10 * np.log10(np.add.reduceat(power, bands, axis=1) + 1e-12)
    return np.column_stack([
        energies - np.median(energies, axis=1, keepdims=True),
        (log - smooth)[:, :896:2],
        (smooth - np.median(smooth, axis=1, keepdims=True))[:, :1536:12],
    ]).astype(np.float32)


def score(model, batch):
    flat = batch.reshape(-1, batch.shape[-1])
    estimator = model["estimator"]
    values = (estimator.decision_function(flat) if model["method"] == "segment_svm"
              else estimator.predict_proba(flat)[:, 1])
    return values.reshape(len(batch), -1).mean(axis=1)


def predict(source, model):
    if hasattr(source, "seek"):
        source.seek(0)
    value = float(score(model, features(source)[None])[0])
    if hasattr(source, "seek"):
        source.seek(0)
    _, audio, times, spectral = display_features(source, display=True)
    return value, value > model["threshold"], audio, times, spectral


def train():
    items = list(recordings(ROOT / "dataset/fan"))
    meta = pd.DataFrame([(p.name, sec, domain, split, label)
                         for p, sec, domain, split, label in items],
                        columns=["file", "section", "domain", "split", "actual"])
    fits, tests = {}, {}
    for section, group in meta.groupby("section"):
        source = group[(group.domain == "source") & (group.split == "test")]
        adapt, holdout = train_test_split(source.index, test_size=.30,
                                          stratify=source.actual, random_state=218)
        normal = group[(group.split == "train") & (group.domain == "source")].sample(
            n=150, random_state=218).index
        fits[section] = np.r_[normal, adapt]
        tests[section] = group.loc[group.index.isin(holdout) |
                                   ((group.domain == "target") & (group.split == "test"))]
    used = sorted(set(np.concatenate([*fits.values(), *[f.index for f in tests.values()]])))
    cache = ROOT / "results/fan_segments/segment_features.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    paths = [items[index][0] for index in used]
    keys = [f"{p}|{p.stat().st_size}|{p.stat().st_mtime_ns}" for p in paths]
    with np.load(cache) if cache.exists() else nullcontext({}) as saved:
        matrix = saved["matrix"] if "matrix" in saved and np.array_equal(saved["keys"], keys) else None
    if matrix is None:
        with ThreadPoolExecutor(max_workers=4) as pool:
            matrix = np.stack(list(pool.map(features, paths)))
        tmp = cache.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, keys=keys, matrix=matrix)
        tmp.replace(cache)
    positions = dict(zip(used, range(len(used))))
    models, predictions = {}, []
    for section, fit in fits.items():
        test = tests[section]
        training = matrix[[positions[i] for i in fit]]
        evaluation = matrix[[positions[i] for i in test.index]]
        for method, estimator, threshold in (
            ("segment_svm", make_pipeline(StandardScaler(), SVC(C=2, class_weight="balanced")), 0.),
            ("segment_trees", ExtraTreesClassifier(n_estimators=240, min_samples_leaf=3,
                              class_weight="balanced", n_jobs=3, random_state=218), .5),
        ):
            estimator.fit(training.reshape(-1, training.shape[-1]), np.repeat(meta.actual.iloc[fit], 5))
            model = dict(method=method, estimator=estimator, threshold=threshold)
            models[section, method] = model
            predictions.append(test.assign(model=method, score=score(model, evaluation),
                                           threshold=threshold))
    report("fan_segments", pd.concat(predictions), ["model", "section", "domain"], models)


if __name__ == "__main__":
    train()
