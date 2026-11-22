
import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
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


def train_temporal_autoencoder():

    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    print("\n========== EXPERIMENT 2: AUTOENCODER ==========")

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

    # Split normal training recordings:
    # 90% model fitting, 10% threshold calibration
    X_fit, X_validation = train_test_split(
        X_train_all,
        test_size=0.10,
        random_state=42
    )

    print("Model fitting samples:", len(X_fit))
    print("Validation samples:", len(X_validation))
    print("Source test samples:", len(X_source))
    print("Target test samples:", len(X_target))

    # Fit scaler using model-fitting data only
    scaler = StandardScaler()

    X_fit = scaler.fit_transform(X_fit)
    X_validation = scaler.transform(X_validation)

    X_source = scaler.transform(X_source)
    X_target = scaler.transform(X_target)

    # Temporal Autoencoder
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 128),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=42
    )

    print("\nTraining temporal Autoencoder...")

    model.fit(X_fit, X_fit)

    print("Training completed!")
    print("Iterations:", model.n_iter_)
    print("Final loss:", model.loss_)

    # Calculate validation reconstruction errors
    validation_reconstructed = model.predict(X_validation)

    validation_errors = np.mean(
        (X_validation - validation_reconstructed) ** 2,
        axis=1
    )

    # Threshold based only on held-out normal data
    threshold = np.percentile(validation_errors, 95)

    print("\nCalibrated threshold:", threshold)

    def evaluate(name, X, y_true):

        reconstructed = model.predict(X)

        errors = np.mean(
            (X - reconstructed) ** 2,
            axis=1
        )

        predictions = (errors > threshold).astype(int)

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

        print("ROC-AUC:", roc_auc_score(y_true, errors))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, predictions))

        return errors, predictions

    source_errors, source_predictions = evaluate(
        "SOURCE DOMAIN",
        X_source,
        y_source
    )

    target_errors, target_predictions = evaluate(
        "TARGET DOMAIN",
        X_target,
        y_target
    )

    # Save separately from Experiment 1
    joblib.dump(
        model,
        MODELS_DIR / "temporal_autoencoder.pkl"
    )

    joblib.dump(
        scaler,
        MODELS_DIR / "temporal_scaler.pkl"
    )

    joblib.dump(
        threshold,
        MODELS_DIR / "temporal_threshold.pkl"
    )

    results = pd.concat([
        pd.DataFrame({
            "filename": source_df["filename"],
            "split": "source_test",
            "actual": y_source,
            "predicted": source_predictions,
            "anomaly_score": source_errors
        }),

        pd.DataFrame({
            "filename": target_df["filename"],
            "split": "target_test",
            "actual": y_target,
            "predicted": target_predictions,
            "anomaly_score": target_errors
        })
    ], ignore_index=True)

    results.to_csv(
        RESULTS_DIR / "temporal_prediction_results.csv",
        index=False
    )

    print("\nExperiment 2 artifacts saved successfully!")


if __name__ == "__main__":
    train_temporal_autoencoder()
