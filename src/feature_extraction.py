
import numpy as np
import pandas as pd
import librosa

from pathlib import Path
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

INVENTORY_PATH = RESULTS_DIR / "dataset_inventory.csv"


def extract_features():

    df = pd.read_csv(INVENTORY_PATH)

    features = []
    metadata = []

    print("\n========== FEATURE EXTRACTION ==========")

    for _, row in tqdm(df.iterrows(), total=len(df)):

        audio, sr = librosa.load(
            row["filepath"],
            sr=None,
            mono=True
        )

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=1024,
            hop_length=256,
            n_mels=64,
            power=2.0
        )

        log_mel = librosa.power_to_db(
            mel,
            ref=1.0
        )

        # Calculate mean and standard deviation
        mean_features = np.mean(log_mel, axis=1)
        std_features = np.std(log_mel, axis=1)

        # Combine both feature vectors
        feature_vector = np.concatenate([
            mean_features,
            std_features
        ])

        features.append(feature_vector)

        metadata.append({
            "filename": row["filename"],
            "split": row["split"],
            "label": row["label"]
        })

    features = np.array(features)

    feature_df = pd.DataFrame(
        features,
        columns=[
            *[f"mel_mean_{i}" for i in range(64)],
            *[f"mel_std_{i}" for i in range(64)]
        ]
    )

    metadata_df = pd.DataFrame(metadata)

    final_df = pd.concat(
        [metadata_df, feature_df],
        axis=1
    )

    output_path = RESULTS_DIR / "extracted_features.csv"

    final_df.to_csv(output_path, index=False)

    print("\nFeature extraction completed!")
    print("Feature shape:", features.shape)
    print("Saved to:", output_path)


if __name__ == "__main__":
    extract_features()
