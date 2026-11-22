
import soundfile as sf
import pandas as pd

from pathlib import Path
from collections import Counter


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset" / "fan"
RESULTS_DIR = BASE_DIR / "results"


def analyze_dataset():

    RESULTS_DIR.mkdir(exist_ok=True)

    if not DATASET_DIR.exists():
        print("Dataset folder not found:", DATASET_DIR)
        return

    audio_files = sorted(DATASET_DIR.rglob("*.wav"))

    print("\n========== DATASET ANALYSIS ==========")
    print("Total audio files:", len(audio_files))

    records = []

    for file in audio_files:

        split = file.relative_to(DATASET_DIR).parts[0]

        filename = file.name.lower()

        if "anomaly" in filename:
            label = "anomaly"
        elif "normal" in filename:
            label = "normal"
        else:
            label = "unknown"

        try:
            info = sf.info(str(file))

            records.append({
                "filename": file.name,
                "filepath": str(file),
                "split": split,
                "label": label,
                "duration": info.duration,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "samples": info.frames
            })

        except Exception as e:
            print("Error reading:", file.name, e)

    df = pd.DataFrame(records)

    print("\n========== FILE DISTRIBUTION ==========")
    print(df["split"].value_counts())

    print("\n========== NORMAL / ANOMALY ==========")
    print(pd.crosstab(df["split"], df["label"]))

    print("\n========== AUDIO INFORMATION ==========")
    print(df[["duration", "sample_rate", "channels"]].describe())

    print("\n========== SAMPLING FREQUENCIES ==========")
    print(df["sample_rate"].value_counts())

    output_path = RESULTS_DIR / "dataset_inventory.csv"

    df.to_csv(output_path, index=False)

    print("\nInventory saved to:", output_path)
    print("\nDataset analysis completed!")


if __name__ == "__main__":
    analyze_dataset()
