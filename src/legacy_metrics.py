"""Recalculate the archived fan experiments on section/domain development splits."""
from pathlib import Path

import pandas as pd

from src.benchmark import ROOT, classification_metrics


def main():
    sources = {"autoencoder": "prediction_results.csv",
               "temporal_autoencoder": "temporal_prediction_results.csv",
               "isolation_forest": "isolation_predictions.csv"}
    rows = []
    for model, file in sources.items():
        frame = pd.read_csv(ROOT / "results" / file)
        frame["section"] = frame.filename.str.extract(r"section_(\d+)")
        for (section, split), group in frame.groupby(["section", "split"]):
            group = group.rename(columns={"anomaly_score": "score"})
            metrics = classification_metrics(group)
            rows.append(dict(model=model, section=section, domain=split.split("_")[0],
                             **metrics))
    output = ROOT / "results" / "fan_legacy_metrics.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(pd.DataFrame(rows).groupby(["model", "domain"])[["roc_auc", "f1"]].mean())


if __name__ == "__main__":
    main()
