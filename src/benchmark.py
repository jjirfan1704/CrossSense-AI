"""Shared benchmark outputs for the dashboard and offline review."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parents[1]


def classification_metrics(frame):
    y, p, scores = frame.actual, frame.predicted, frame.score
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, accuracy=accuracy_score(y, p),
                balanced_accuracy=balanced_accuracy_score(y, p),
                average_precision=average_precision_score(y, scores),
                precision=precision_score(y, p, zero_division=0),
                recall=recall_score(y, p, zero_division=0),
                f1=f1_score(y, p, zero_division=0),
                roc_auc=roc_auc_score(y, scores),
                standardized_pauc_10=roc_auc_score(y, scores, max_fpr=.1))


def save_figures(machine, predictions, curves, metrics):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    folder = ROOT / "graphs" / machine
    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, frame in predictions.groupby("model"):
        x = pd.to_datetime(frame.timestamp) if machine.startswith("bearing") else range(len(frame))
        ax.plot(x, frame.score - frame.threshold, label=model, alpha=.75)
    if machine.startswith("bearing"):
        ax.set_yscale("symlog")
        fig.autofmt_xdate()
    ax.axhline(0, linewidth=1, color="black")
    ax.set(ylabel="Score minus threshold", xlabel="Timestamp" if machine.startswith("bearing") else "Evaluation sample")
    ax.legend()
    fig.savefig(folder / "scores.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for tags, frame in curves:
        ax.plot(frame.fpr, frame.tpr, label=" / ".join(map(str, tags)))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(fontsize="x-small", ncol=2)
    fig.savefig(folder / "roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots((len(metrics) + 2) // 3, 3, squeeze=False,
                             figsize=(10, 2.8 * ((len(metrics) + 2) // 3)))
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, (tags, _), row in zip(axes.flat, curves, [r for r in metrics if "tn" in r]):
        ax.set_visible(True)
        matrix = [[row["tn"], row["fp"]], [row["fn"], row["tp"]]]
        ax.imshow(matrix, cmap="Blues")
        for y, values in enumerate(matrix):
            for x, value in enumerate(values):
                ax.text(x, y, value, ha="center", va="center", color="black",
                        bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
        ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="Actual",
               title=" / ".join(map(str, tags)))
    fig.tight_layout()
    fig.savefig(folder / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def report(machine, predictions, grouping, models):
    folder = ROOT / "results" / machine
    folder.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / "models" / machine
    model_dir.mkdir(parents=True, exist_ok=True)
    predictions = predictions.copy()
    predictions["predicted"] = (predictions.score > predictions.threshold).astype(int)
    metrics, curves = [], []
    for keys, frame in predictions.groupby(grouping):
        tags = dict(zip(grouping, keys))
        labels = frame.dropna(subset=["actual"])
        values = dict(tags, samples=len(frame), labeled=len(labels),
                      alarm_rate=frame.predicted.mean())
        if labels.actual.nunique() == 2:
            values.update(classification_metrics(labels))
            fpr, tpr, _ = roc_curve(labels.actual, labels.score)
            curves.append((keys, pd.DataFrame(dict(tags, fpr=fpr, tpr=tpr))))
        if len(labels) and labels.actual.eq(1).all():
            values["late_proxy_detection_rate"] = labels.predicted.mean()
            persistent = frame[frame.persistent_alarm]
            if len(persistent):
                values["first_three_consecutive_alarm"] = persistent.timestamp.iloc[0]
        metrics.append(values)
    predictions.to_csv(folder / "predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(folder / "metrics.csv", index=False)
    if curves:
        pd.concat([curve for _, curve in curves]).to_csv(folder / "roc.csv", index=False)
    save_figures(machine, predictions, curves, metrics)
    joblib.dump(models, model_dir / "benchmark.joblib", compress=3)
    print(pd.DataFrame(metrics).to_string(index=False))
