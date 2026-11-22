
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc
)

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"
GRAPHS_DIR = BASE_DIR / "graphs"


def generate_evaluation_graphs():

    df = pd.read_csv(
        RESULTS_DIR / "prediction_results.csv"
    )

    GRAPHS_DIR.mkdir(exist_ok=True)

    # 1. Anomaly score distributions
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, split in zip(
        axes, ["source_test", "target_test"]
    ):

        subset = df[df["split"] == split]

        normal = subset[
            subset["actual"] == 0
        ]["anomaly_score"]

        anomaly = subset[
            subset["actual"] == 1
        ]["anomaly_score"]

        ax.hist(
            normal,
            bins=30,
            alpha=0.6,
            label="Normal"
        )

        ax.hist(
            anomaly,
            bins=30,
            alpha=0.6,
            label="Anomaly"
        )

        ax.set_title(split.replace("_", " ").title())
        ax.set_xlabel("Reconstruction Error")
        ax.set_ylabel("Number of Samples")
        ax.legend()

    plt.tight_layout()

    plt.savefig(
        GRAPHS_DIR / "anomaly_score_distribution.png",
        dpi=300
    )

    plt.show()

    # 2. Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, split in zip(
        axes, ["source_test", "target_test"]
    ):

        subset = df[df["split"] == split]

        cm = confusion_matrix(
            subset["actual"],
            subset["predicted"]
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"],
            ax=ax
        )

        ax.set_title(split.replace("_", " ").title())
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()

    plt.savefig(
        GRAPHS_DIR / "confusion_matrices.png",
        dpi=300
    )

    plt.show()

    # 3. ROC curves
    plt.figure(figsize=(7, 6))

    for split in ["source_test", "target_test"]:

        subset = df[df["split"] == split]

        fpr, tpr, _ = roc_curve(
            subset["actual"],
            subset["anomaly_score"]
        )

        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{split}: AUC = {roc_auc:.3f}"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - CrossSense AI")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        GRAPHS_DIR / "roc_curve.png",
        dpi=300
    )

    plt.show()

    print("All evaluation graphs generated!")


if __name__ == "__main__":
    generate_evaluation_graphs()
