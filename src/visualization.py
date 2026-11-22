
from email.mime import audio

import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
GRAPHS_DIR = BASE_DIR / "graphs"

INVENTORY_PATH = RESULTS_DIR / "dataset_inventory.csv"


def visualize_audio():

    GRAPHS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(INVENTORY_PATH)

    # Select one normal training sample
    normal_sample = df[
        (df["split"] == "train") &
        (df["label"] == "normal")
    ].iloc[0]

    # Select one anomalous source-test sample
    anomaly_sample = df[
        (df["split"] == "source_test") &
        (df["label"] == "anomaly")
    ].iloc[0]

    samples = [
        ("Normal Training Audio", normal_sample),
        ("Anomalous Source Audio", anomaly_sample)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    for i, (title, sample) in enumerate(samples):

        audio_path = sample["filepath"]

        # Load audio without changing sample rate
        audio, sr = librosa.load(
            audio_path,
            sr=None,
            mono=True
        )

    
        # Time axis
        import numpy as np
        time = np.arange(len(audio)) / sr

        # Waveform
        axes[i, 0].plot(time, audio)

        axes[i, 0].set_title(title + " - Waveform")
        axes[i, 0].set_xlabel("Time (seconds)")
        axes[i, 0].set_ylabel("Amplitude")

        # Generate Mel Spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=1024,
            hop_length=256,
            n_mels=64
        )

        # Convert power to decibels
        mel_db = librosa.power_to_db(
            mel,
            ref=mel.max()
        )

        # Display spectrogram
        img = librosa.display.specshow(
            mel_db,
            sr=sr,
            hop_length=256,
            x_axis="time",
            y_axis="mel",
            ax=axes[i, 1]
        )

        axes[i, 1].set_title(title + " - Log-Mel Spectrogram")

    fig.colorbar(
    img,
    ax=axes[:, 1].tolist(),
    format="%+2.0f dB",
    pad=0.03
    )

    fig.subplots_adjust(
        hspace=0.45,
        wspace=0.30,
        right=0.88
    )

    output_path = GRAPHS_DIR / "audio_visualization.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("\nVisualization completed!")
    print("Saved at:", output_path)


if __name__ == "__main__":
    visualize_audio()
