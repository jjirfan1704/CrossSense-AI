"""Benchmark the full MSM archive using nested, run-disjoint evaluation."""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

from src.benchmark import ROOT, classification_metrics, report
from src.cnc_prepare import CACHE

MODELS = {"random_forest": RandomForestClassifier, "extra_trees": ExtraTreesClassifier}


def group_folds(x, y, groups, max_folds, seed):
    counts = [len(set(groups[y == label])) for label in (0, 1)]
    for size in range(min(max_folds, *counts), 1, -1):
        for offset in range(15):
            folds = list(StratifiedGroupKFold(size, shuffle=True, random_state=seed + offset)
                         .split(x, y, groups))
            if all(len(np.unique(y[fit])) == 2 for fit, _ in folds):
                return folds
    return []


def operating_threshold(y, scores):
    fpr, recall, cut = roc_curve(y, scores, drop_intermediate=False)
    allowed = (recall >= .9) & np.isfinite(cut)
    if allowed.any():
        index = np.flatnonzero(allowed)[np.argmin(fpr[allowed])]
        return float(np.nextafter(cut[index], -np.inf))
    return .5


def sensor_scores(estimator, x, y, fit, test_values):
    fit = fit[np.isfinite(x[fit]).all(axis=1)]
    valid = np.isfinite(test_values).all(axis=1)
    scores = np.full(len(test_values), np.nan)
    if np.unique(y[fit]).size < 2:
        return scores, False
    estimator.fit(x[fit], y[fit])
    if valid.any():
        scores[valid] = estimator.predict_proba(test_values[valid])[:, 1]
    return scores, True


def blend(matrix, weights):
    available = np.isfinite(matrix)
    safe = np.nan_to_num(matrix)
    effective = available * weights[:, None]
    denominator = effective.sum(axis=0)
    # Use the available sensor mean when every positively weighted sensor is missing.
    fallback = denominator == 0
    effective[:, fallback] = available[:, fallback]
    denominator = effective.sum(axis=0)
    return np.divide((safe * effective).sum(axis=0), denominator,
                     out=np.full(matrix.shape[1], np.nan), where=denominator > 0)


def validation_auc(y, values):
    use = np.isfinite(values)
    return roc_auc_score(y[use], values[use]) if np.unique(y[use]).size == 2 else -1


def fit_models(train, truth, splits, schemas, test=None):
    predictions, validation, estimators = {}, {}, {}
    for sensor, columns in schemas.items():
        x = train[columns].to_numpy()
        holdout = test[columns].to_numpy() if test is not None else x[:0]
        for name, model in MODELS.items():
            key = f"{sensor}/{name}"
            estimator = model(n_estimators=72, min_samples_leaf=2, class_weight="balanced",
                              random_state=42, n_jobs=2)
            if splits:
                scores = np.full(len(train), np.nan)
                for fit, val in splits:
                    scores[val], _ = sensor_scores(estimator, x, truth, fit, x[val])
                validation[key] = scores
            values, fitted = sensor_scores(estimator, x, truth, np.arange(len(x)), holdout)
            if fitted:
                estimators[key] = estimator
            if test is not None:
                predictions[key] = values
    weights = {}
    keys = [max((key for key in estimators if key.startswith(f"{sensor}/")),
                key=lambda key: validation_auc(truth, validation[key]) if splits else 0,
                default=None) for sensor in schemas]
    keys = [key for key in keys if key is not None]
    if keys:
        equal = np.full(len(keys), 1 / len(keys))
        best = equal
        if splits:
            matrix = np.stack([validation[key] for key in keys])
            candidates = [equal, *[.5 * equal + .5 * unit for unit in np.eye(len(keys))],
                          *np.eye(len(keys))]
            best = max(candidates, key=lambda w: validation_auc(truth, blend(matrix, w)))
            validation["equal_fusion"] = blend(matrix, equal)
            validation["validation_fusion"] = blend(matrix, best)
        weights = dict(zip(keys, map(float, best)))
        if test is not None:
            matrix = np.stack([predictions[key] for key in keys])
            predictions["equal_fusion"] = blend(matrix, equal)
            predictions["validation_fusion"] = blend(matrix, best)
    thresholds = {}
    for key, values in validation.items():
        valid = np.isfinite(values)
        if np.unique(truth[valid]).size == 2:
            thresholds[key] = operating_threshold(truth[valid], values[valid])
    return estimators, predictions, thresholds, weights


def predict_features(row, pack, method):
    if method == "validation_fusion":
        weights = pack["weights"]
    else:
        weights = {method: 1.0}
    values = {}
    for key in weights:
        columns = pack["schemas"][key.split("/")[0]]
        x = row[columns].to_numpy(dtype=float).reshape(1, -1)
        values[key] = (float(pack["estimators"][key].predict_proba(x)[0, 1])
                       if np.isfinite(x).all() else np.nan)
    score = float(blend(np.array(list(values.values()))[:, None], np.array(list(weights.values())))[0])
    if not np.isfinite(score):
        raise ValueError("No usable sensor for the selected interval and detector.")
    return score, score > pack["thresholds"].get(method, .5), values


def evaluate(path):
    raw = pd.read_csv(path)
    results, diagnostics, models = [], [], {}
    for machine, group in raw.groupby("machine"):
        binary = group[group.actual.isin([0, 1])].copy()
        features = [column for column in binary if ":" in column and binary[column].notna().any()]
        schemas = {}
        for column in features:
            schemas.setdefault(column.split(":")[0], []).append(column)
        available = [binary[columns].notna().all(axis=1) for columns in schemas.values()]
        usable = np.logical_or.reduce(available) if available else np.zeros(len(binary), dtype=bool)
        rows = binary[usable].reset_index(drop=True)
        y, groups = rows.actual.to_numpy(dtype=int), rows.run.to_numpy()
        counts = [int((y == label).sum()) for label in (0, 1)]
        positives = len(set(groups[y == 1]))
        diagnostic = dict(machine=machine, total_runs=group.run.nunique(),
                          scored_runs=rows.run.nunique(), labeled_binary=len(binary),
                          scorable_binary=len(rows), normal=counts[0], anomaly=counts[1],
                          positive_runs=positives, sensors=", ".join(schemas))
        outer = group_folds(rows, y, groups, 5, 42)
        if not outer:
            diagnostic["status"] = "Insufficient class support for run-disjoint evaluation"
            diagnostics.append(diagnostic)
            continue
        for fold, (fit, test) in enumerate(outer, start=1):
            train, val = rows.iloc[fit], rows.iloc[test]
            training_labels = y[fit]
            inner = group_folds(train, training_labels, groups[fit], 3, 100)
            _, predictions, thresholds, weights = fit_models(train, training_labels, inner, schemas, val)
            predictions["always_normal"] = np.zeros(len(val))
            for key, values in predictions.items():
                result = val[["machine", "run", "start", "end", "actual"]].copy()
                result = result.assign(fold=fold, model=key, score=values,
                                       threshold=thresholds.get(key, .5))
                results.append(result[result.score.notna()])
            print(f"{machine} fold {fold}/{len(outer)}: {len(val)} intervals, "
                  f"{len(set(groups[test]))} test runs, inner validation={bool(inner)}", flush=True)
        estimators, _, thresholds, weights = fit_models(rows, y, outer, schemas)
        models[machine] = dict(estimators=estimators, thresholds=thresholds,
                               schemas=schemas, weights=weights)
        diagnostic["status"] = f"{len(outer)} held-out run folds"
        diagnostics.append(diagnostic)
    folder = ROOT / "results" / "cnc_full"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(diagnostics).to_csv(folder / "coverage.csv", index=False)
    if not results:
        raise ValueError("No machine has enough run-disjoint labels to evaluate")
    predictions = pd.concat(results, ignore_index=True)
    report("cnc_full", predictions, ["machine", "model"], models)
    matched = []
    for machine, group in predictions.groupby("machine"):
        keys = ["run", "start", "end", "actual"]
        availability = group.groupby(keys).model.transform("nunique")
        comparable = group[availability.eq(group.model.nunique())].copy()
        comparable["predicted"] = (comparable.score > comparable.threshold).astype(int)
        for key, subset in comparable.groupby("model"):
            if subset.actual.nunique() == 2:
                matched.append(dict(machine=machine, model=key, samples=len(subset),
                                    **classification_metrics(subset)))
    pd.DataFrame(matched).to_csv(folder / "matched_metrics.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=CACHE)
    evaluate(parser.parse_args().features)

