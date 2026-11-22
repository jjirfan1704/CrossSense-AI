"""MSM cutting-interval benchmark for synchronized sensor channels."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from src.benchmark import ROOT, report

MODALITIES = ("acc", "ct", "mtc")
MTC_CHANNELS = ("watt", "smarms", "xmarms", "ymarms", "zmarms")


def read_sensor(source, modality=None):
    frame = pd.read_csv(source)
    if frame.shape[1] < 2:
        raise ValueError("Sensor CSV needs a time column followed by numeric channels.")
    if modality == "mtc":
        channels = [name for name in MTC_CHANNELS if name in frame]
        if not channels:
            raise ValueError("MTConnect CSV has no supported power or vibration channels.")
        frame = frame.loc[:, [frame.columns[0], *channels]]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=[frame.columns[0]]).set_index(frame.columns[0]).sort_index()
    if frame.empty or not np.isfinite(frame.index).all():
        raise ValueError("Sensor CSV has no valid time axis.")
    return frame


def source_modality(name):
    return "mtc" if name.startswith("mtc_") else name


def sensor_views(sensors):
    views = {name: frame for name, frame in sensors.items() if name != "mtc"}
    if "mtc" in sensors:
        frame = sensors["mtc"]
        if "watt" in frame:
            views["mtc_power"] = frame.loc[:, ["watt"]]
        vibration = [column for column in frame if column != "watt"]
        if vibration:
            views["mtc_vibration"] = frame.loc[:, vibration]
    return views


def interval_features(frame, start, end, columns):
    left, right = frame.index.searchsorted([start, end])
    window = frame.iloc[left:right].loc[:, columns]
    if len(window) < 8 or not np.isfinite(window.to_numpy()).all():
        return None
    values = window.to_numpy()
    return np.r_[values.mean(0), values.std(0), np.sqrt((values ** 2).mean(0)),
                 np.ptp(values, axis=0), np.quantile(values, .9, axis=0)]


def train(root):
    root = Path(root)
    runs = sorted(root.rglob("label.csv"))
    if not runs:
        raise ValueError("No MSM label.csv files found. Choose a single extracted machine folder.")
    schemas, features, records = {}, {}, []
    machine_folders = {path.parent.parent for path in runs}
    if len(machine_folders) != 1:
        raise ValueError("Choose one MSM machine folder with matching sensor configurations.")
    for label_path in runs:
        sensors = {}
        for name in MODALITIES:
            files = sorted(label_path.parent.glob(f"*_{name}.csv"))
            if len(files) == 1:
                try:
                    sensors[name] = read_sensor(files[0], name)
                except ValueError as error:
                    print(f"Skipping {files[0]}: {error}")
        sensors = sensor_views(sensors)
        if not sensors:
            continue
        if not schemas:
            schemas = {name: list(frame.columns) for name, frame in sensors.items()}
            features = {name: [] for name in schemas}
        if set(schemas) != set(sensors) or any(list(sensors[n].columns) != c for n, c in schemas.items()):
            continue
        labels = pd.read_csv(label_path)
        labels.columns = labels.columns.str.strip().str.lower()
        for start, end, label in labels[["start", "end", "label"]].itertuples(index=False, name=None):
            if label not in (0, 1) or not np.isfinite([start, end]).all() or end <= start:
                continue
            interval = {n: interval_features(sensors[n], start, end, c) for n, c in schemas.items()}
            if any(value is None for value in interval.values()):
                continue
            records.append((str(label_path.parent.relative_to(root)), start, end, int(label)))
            for name, value in interval.items():
                features[name].append(value)
    meta = pd.DataFrame(records, columns=["run", "start", "end", "actual"])
    if meta.run.nunique() < 5 or meta.actual.nunique() != 2:
        raise ValueError("Need at least five compatible runs and both stable and abnormal cutting intervals.")
    for fit, test in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(meta, meta.actual, meta.run):
        if meta.actual.iloc[fit].nunique() == meta.actual.iloc[test].nunique() == 2:
            break
    else:
        raise ValueError("Could not obtain disjoint training/test runs with both classes.")
    models, scores, validation = {}, {}, {}
    train_labels = meta.actual.iloc[fit]
    train_runs = meta.run.iloc[fit]
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=43).split(
        meta.iloc[fit], train_labels, train_runs
    ))
    can_fit_weights = len(features) > 1 and all(
        train_labels.iloc[train].nunique() == 2 for train, _ in folds
    )
    for name, values in features.items():
        x = np.stack(values)
        estimator = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                           class_weight="balanced", random_state=42)
        if can_fit_weights:
            validation[name] = cross_val_predict(estimator, x[fit], train_labels,
                                                 cv=folds, method="predict_proba")[:, 1]
        estimator.fit(x[fit], train_labels)
        scores[name] = estimator.predict_proba(x[test])[:, 1]
        models[name] = estimator
    weights = dict.fromkeys(scores, 1 / len(scores))
    if can_fit_weights:
        n = len(validation)
        equal = np.full(n, 1 / n)
        candidates = [equal, *[.5 * equal + .5 * unit for unit in np.eye(n)], *np.eye(n)]
        validation_scores = np.vstack(list(validation.values()))
        best = max(candidates, key=lambda w: roc_auc_score(train_labels, w @ validation_scores))
        weights = dict(zip(validation, map(float, best)))
    if len(scores) > 1:
        equal_scores = np.mean(list(scores.values()), axis=0)
        scores["fusion"] = sum(weights[name] * values for name, values in scores.items())
        scores["equal_fusion"] = equal_scores
    results = [meta.iloc[test].assign(model=name, score=value, threshold=.5)
               for name, value in scores.items()]
    absolute = root.resolve()
    data_root = absolute.relative_to(ROOT) if absolute.is_relative_to(ROOT) else absolute
    report("cnc", pd.concat(results), ["model"],
           dict(estimators=models, schemas=schemas, weights=weights,
                machine=next(iter(machine_folders)).name, data_root=data_root))


def predict(sensors, start, end, model):
    if not np.isfinite([start, end]).all() or end <= start:
        raise ValueError("Choose a cutting interval with start before end.")
    scores = {}
    for name, columns in model["schemas"].items():
        frame = sensors[source_modality(name)]
        if not set(columns).issubset(frame.columns):
            raise ValueError(f"{name} CSV is missing required channels.")
        x = interval_features(frame, start, end, columns)
        if x is None:
            raise ValueError(f"{name} needs at least eight finite samples in the interval.")
        scores[name] = float(model["estimators"][name].predict_proba(x.reshape(1, -1))[0, 1])
    value = sum(model["weights"][name] * value for name, value in scores.items())
    return scores, value, value > .5




