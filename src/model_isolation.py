
import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

FEATURE_PATH = RESULTS_DIR / "temporal_features.csv"


def train_isolation_forest():

    print("\n========== EXPERIMENT 3: ISOLATION FOREST ==========")

    df = pd.read_csv(FEATURE_PATH)

    feature_columns = [
        col for col in df.columns
        if col.startswith("segment_")
    ]

    train_df = df[df["split"] == "train"].copy()
    source_df = df[df["split"] == "source_test"].copy()
    target_df = df[df["split"] == "target_test"].copy()

    X_train_all = train_df[feature_columns].values
    X_source = source_df[feature_columns].values
    X_target = target_df[feature_columns].values

    y_source = (
        source_df["label"] == "anomaly"
    ).astype(int).values

    y_target = (
        target_df["label"] == "anomaly"
    ).astype(int).values

    X_fit, X_validation = train_test_split(
        X_train_all,
        test_size=0.10,
        random_state=42
    )

    scaler = StandardScaler()

    X_fit = scaler.fit_transform(X_fit)
    X_validation = scaler.transform(X_validation)

    X_source = scaler.transform(X_source)
    X_target = scaler.transform(X_target)

    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    print("Training samples:", len(X_fit))
    print("Validation samples:", len(X_validation))

    print("\nTraining Isolation Forest...")

    model.fit(X_fit)

    print("Training completed!")

    # Higher values indicate more anomalous samples
    validation_scores = -model.score_samples(X_validation)

    # Threshold based only on normal validation data
    threshold = np.percentile(validation_scores, 95)

    print("\nAnomaly threshold:", threshold)

    def evaluate(name, X, y_true):

        scores = -model.score_samples(X)

        predictions = (
            scores > threshold
        ).astype(int)

        print(f"\n========== {name} RESULTS ==========")

        print("Accuracy:", accuracy_score(y_true, predictions))

        print("Precision:", precision_score(
            y_true, predictions, zero_division=0
        ))

        print("Recall:", recall_score(
            y_true, predictions, zero_division=0
        ))

        print("F1 Score:", f1_score(
            y_true, predictions, zero_division=0
        ))

        print("ROC-AUC:", roc_auc_score(y_true, scores))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, predictions))

        return scores, predictions

    source_scores, source_predictions = evaluate(
        "SOURCE DOMAIN",
        X_source,
        y_source
    )

    target_scores, target_predictions = evaluate(
        "TARGET DOMAIN",
        X_target,
        y_target
    )

    MODELS_DIR.mkdir(exist_ok=True)

    joblib.dump(
        model,
        MODELS_DIR / "isolation_forest.pkl"
    )

    joblib.dump(
        scaler,
        MODELS_DIR / "isolation_scaler.pkl"
    )

    joblib.dump(
        threshold,
        MODELS_DIR / "isolation_threshold.pkl"
    )

    results = pd.concat([
        pd.DataFrame({
            "filename": source_df["filename"],
            "split": "source_test",
            "actual": y_source,
            "predicted": source_predictions,
            "anomaly_score": source_scores
        }),
        pd.DataFrame({
            "filename": target_df["filename"],
            "split": "target_test",
            "actual": y_target,
            "predicted": target_predictions,
            "anomaly_score": target_scores
        })
    ], ignore_index=True)

    results.to_csv(
        RESULTS_DIR / "isolation_predictions.csv",
        index=False
    )

    print("\nExperiment 3 saved successfully!")


if __name__ == "__main__":
    train_isolation_forest()
