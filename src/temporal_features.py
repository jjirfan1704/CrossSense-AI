
import numpy as np
import pandas as pd
import librosa

from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"
INVENTORY_PATH = RESULTS_DIR / "dataset_inventory.csv"


def extract_temporal_features():

    df = pd.read_csv(INVENTORY_PATH)

    all_features = []
    metadata = []

    print("\n========== EXPERIMENT 2 ==========")
    print("Temporal Log-Mel Feature Extraction")

    for _, row in tqdm(df.iterrows(), total=len(df)):

        audio, sr = librosa.load(
            row["filepath"],
            sr=None,
            mono=True
        )

        segment_length = sr  # 1 second

        recording_features = []

        for start in range(0, len(audio), segment_length):

            segment = audio[start:start + segment_length]

            if len(segment) != segment_length:
                continue

            mel = librosa.feature.melspectrogram(
                y=segment,
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

            # Average each Mel frequency band
            segment_features = np.mean(
                log_mel,
                axis=1
            )

            recording_features.extend(segment_features)

        all_features.append(recording_features)

        metadata.append({
            "filename": row["filename"],
            "filepath": row["filepath"],
            "split": row["split"],
            "label": row["label"]
        })

    feature_array = np.array(all_features)

    print("\nFeature array shape:", feature_array.shape)

    feature_columns = [
        f"segment_{segment}_mel_{mel}"
        for segment in range(10)
        for mel in range(64)
    ]

    feature_df = pd.DataFrame(
        feature_array,
        columns=feature_columns
    )

    metadata_df = pd.DataFrame(metadata)

    final_df = pd.concat(
        [metadata_df, feature_df],
        axis=1
    )

    output_path = RESULTS_DIR / "temporal_features.csv"

    final_df.to_csv(output_path, index=False)

    print("\nTemporal feature extraction completed!")
    print("Total recordings:", len(final_df))
    print("Features per recording:", feature_array.shape[1])
    print("Saved to:", output_path)


if __name__ == "__main__":
    extract_temporal_features()
