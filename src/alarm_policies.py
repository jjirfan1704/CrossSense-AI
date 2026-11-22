"""Evaluate causal bearing alarms and an industrial CNC missing-sensor policy."""
import numpy as np
import pandas as pd

from src.benchmark import ROOT, classification_metrics


def bearing_alarm(scores, threshold, window=3, votes=2):
    flags = pd.Series(np.asarray(scores) > np.asarray(threshold))
    return flags.rolling(window, min_periods=window).sum().ge(votes).to_numpy()


def bearing():
    data = pd.read_csv(ROOT / "results/bearing_reference/predictions.csv")
    low = data[data.model.eq("low_band")].sort_values("timestamp").copy()
    rows = []
    for window, votes in [(1, 1), (3, 2), (5, 3), (5, 4)]:
        alarms = bearing_alarm(low.score, low.threshold, window, votes)
        rows.append(dict(window=window, votes=votes, samples=len(alarms),
                         alarm_rate=alarms.mean(),
                         proxy_snapshots=int(low.actual.eq(1).sum()),
                         late_flags=int(alarms[low.actual.eq(1)].sum()),
                         first_alarm=low.loc[alarms, "timestamp"].min()))
        low[f"alarm_{votes}_of_{window}"] = alarms
    low.to_csv(ROOT / "results/bearing_reference/persistence_predictions.csv", index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(ROOT / "results/bearing_reference/persistence_metrics.csv", index=False)
    print(metrics.to_string(index=False))


def cnc():
    data = pd.read_csv(ROOT / "results/cnc_full/predictions.csv", float_precision="round_trip")
    data = data[data.machine.eq("tmf_vf10")]
    index = ["run", "start", "end", "actual", "fold"]
    primary = data[data.model.eq("mtc_power/random_forest")].set_index(index)
    fallback = data[data.model.eq("s0/random_forest")].set_index(index)
    policy = primary.reindex(fallback.index).combine_first(fallback).reset_index()
    policy["detector"] = np.where(policy.model.eq("mtc_power/random_forest"),
                                  "Controller power", "Microphone S0 fallback")
    policy["predicted"] = policy.predicted.astype(int)
    columns = ["accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"]
    result = {"power_then_s0": classification_metrics(policy)}
    for name in ("validation_fusion", "equal_fusion"):
        result[name] = classification_metrics(data[data.model.eq(name)])
    metrics = pd.DataFrame(result).T[columns]
    metrics.index.name = "policy"
    folder = ROOT / "results/cnc_full"
    policy.to_csv(folder / "industrial_fallback_predictions.csv", index=False)
    metrics.to_csv(folder / "industrial_fallback_metrics.csv")
    print(metrics.to_string())


if __name__ == "__main__":
    bearing()
    cnc()
