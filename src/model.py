
import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
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

FEATURE_PATH = RESULTS_DIR / "extracted_features.csv"


def train_autoencoder():

    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    print("\n========== AUTOENCODER TRAINING ==========")

    # Load extracted features
    df = pd.read_csv(FEATURE_PATH)

    feature_columns = [
        col for col in df.columns
        if col.startswith("mel_mean_")
        or col.startswith("mel_std_")
    ]

    # Separate training and testing data
    train_df = df[df["split"] == "train"].copy()

    source_df = df[df["split"] == "source_test"].copy()

    target_df = df[df["split"] == "target_test"].copy()

    X_train = train_df[feature_columns].values
    X_source = source_df[feature_columns].values
    X_target = target_df[feature_columns].values

    y_source = (
        source_df["label"] == "anomaly"
    ).astype(int).values

    y_target = (
        target_df["label"] == "anomaly"
    ).astype(int).values

    print("Training samples:", len(X_train))
    print("Source test samples:", len(X_source))
    print("Target test samples:", len(X_target))

    # Normalize features using training data only
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_source = scaler.transform(X_source)
    X_target = scaler.transform(X_target)

    # Build neural autoencoder
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42
    )

    print("\nTraining Autoencoder...")

    # Train using normal data only
    model.fit(X_train, X_train)

    print("Training completed!")

    print("Iterations:", model.n_iter_)
    print("Final training loss:", model.loss_)

    # Reconstruction error
    train_reconstructed = model.predict(X_train)

    train_errors = np.mean(
        (X_train - train_reconstructed) ** 2,
        axis=1
    )

    # Set threshold using normal training errors
    threshold = np.percentile(train_errors, 95)

    print("\nAnomaly threshold:", threshold)

    # Evaluate source domain
    source_reconstructed = model.predict(X_source)

    source_errors = np.mean(
        (X_source - source_reconstructed) ** 2,
        axis=1
    )

    source_predictions = (
        source_errors > threshold
    ).astype(int)

    # Evaluate target domain
    target_reconstructed = model.predict(X_target)

    target_errors = np.mean(
        (X_target - target_reconstructed) ** 2,
        axis=1
    )

    target_predictions = (
        target_errors > threshold
    ).astype(int)

    # Evaluation function
    def evaluate(name, y_true, y_pred, scores):

        print(f"\n========== {name} RESULTS ==========")

        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("Precision:", precision_score(
            y_true, y_pred, zero_division=0
        ))
        print("Recall:", recall_score(
            y_true, y_pred, zero_division=0
        ))
        print("F1 Score:", f1_score(
            y_true, y_pred, zero_division=0
        ))
        print("ROC-AUC:", roc_auc_score(y_true, scores))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

    evaluate(
        "SOURCE DOMAIN",
        y_source,
        source_predictions,
        source_errors
    )

    evaluate(
        "TARGET DOMAIN",
        y_target,
        target_predictions,
        target_errors
    )

    # Save model and scaler
    joblib.dump(model, MODELS_DIR / "autoencoder.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    joblib.dump(
        threshold,
        MODELS_DIR / "threshold.pkl"
    )

    # Save predictions
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
        RESULTS_DIR / "prediction_results.csv",
        index=False
    )

    print("\nModel saved successfully!")
    print("Prediction results saved successfully!")


if __name__ == "__main__":
    train_autoencoder()
