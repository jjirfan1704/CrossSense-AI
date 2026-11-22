"""Prepare fixed interval features directly from the verified MSM archive."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import welch

from src.benchmark import ROOT
from src.cnc import interval_features, read_sensor, sensor_views

CACHE = ROOT / "results" / "cnc_full" / "features.csv"
INDUSTRIAL = ["s_load", "xpw", "ypw", "zpw"]


def sample_features(values):
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < 8:
        return None
    return np.r_[values.mean(0), values.std(0), np.sqrt((values ** 2).mean(0)),
                 np.ptp(values, axis=0), np.quantile(values, .9, axis=0)]


def csv_windows(archive, file, intervals):
    parts = [[] for _ in intervals]
    offset = 0
    with archive.open(file) as source:
        for chunk in pd.read_csv(source, chunksize=250_000, dtype=np.float64):
            step = 32 if file.endswith("_acc.csv") else 1
            sampled = chunk.iloc[(-offset) % step::step].to_numpy()
            offset += len(chunk)
            t, values = sampled[:, 0], sampled[:, 1:]
            for piece, (start, end) in zip(parts, intervals):
                selected = values[(start <= t) & (t < end)]
                if len(selected):
                    piece.append(selected)
    return [sample_features(np.concatenate(piece)) if piece else None for piece in parts]


def audio_windows(archive, file, intervals):
    with sf.SoundFile(BytesIO(archive.read(file))) as reader:
        rate = reader.samplerate
        stride = max(1, rate // 16000)
        features = []
        for start, end in intervals:
            left, right = round(start * rate), min(len(reader), round(end * rate))
            if left < 0 or left >= right:
                features.append(None)
                continue
            reader.seek(left)
            audio = reader.read(right - left, dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            stats = sample_features(audio[::stride, None])
            if stats is None:
                features.append(None)
                continue
            # Welch features retain the original sample rate and avoid decimation aliasing.
            hz, power = welch(audio, rate, nperseg=min(2048, len(audio)))
            bands = np.r_[0, np.geomspace(80, min(8000, rate / 2), 14)]
            energy = np.array([power[(hz >= a) & (hz < b)].sum()
                               for a, b in zip(bands, bands[1:])])
            features.append(np.r_[stats, np.log10(energy + 1e-12)])
        return features


def prepare_run(path, machine, run, files, checkpoints):
    checkpoint = checkpoints / f"{machine}-{run}.csv"
    if checkpoint.is_file():
        return checkpoint
    rows = []
    with ZipFile(path) as archive:
        if "label.csv" not in files:
            raise ValueError(f"Missing annotations for {machine}/{run}")
        with archive.open(files["label.csv"]) as source:
            labels = pd.read_csv(source)
        labels.columns = labels.columns.str.strip().str.lower()
        records = labels[["start", "end", "label"]].apply(pd.to_numeric, errors="coerce")
        records = records.dropna()
        records = records[records.label.isin([0, 1, 2]) & (records.end > records.start)]
        intervals = list(records[["start", "end"]].itertuples(index=False, name=None))
        extracted = {}
        for kind in ("acc.csv", "ct.csv", "s0.wav", "s1.wav", "s2.wav"):
            if kind in files:
                extract = audio_windows if kind.endswith(".wav") else csv_windows
                extracted[kind.split(".")[0]] = extract(archive, files[kind], intervals)
        with archive.open(files["mtc.csv"]) as source:
            if machine == "tmf_vf10":
                frame = read_sensor(source)
                views = {"mtc_load": frame[INDUSTRIAL[:1]], "mtc_power": frame[INDUSTRIAL[1:]]}
            else:
                views = sensor_views({"mtc": read_sensor(source, "mtc")})
        for name, frame in views.items():
            extracted[name] = [interval_features(frame, a, b, list(frame.columns))
                               for a, b in intervals]
        for idx, (start, end, label) in enumerate(records.itertuples(index=False, name=None)):
            row = dict(machine=machine, run=run, start=start, end=end, actual=int(label))
            for name, values in extracted.items():
                if values[idx] is not None:
                    row.update({f"{name}:{i}": value for i, value in enumerate(values[idx])})
            rows.append(row)
    pd.DataFrame(rows, columns=None if rows else ["machine", "run", "start", "end", "actual"]).to_csv(
        checkpoint.with_suffix(".tmp"), index=False)
    checkpoint.with_suffix(".tmp").replace(checkpoint)
    print(f"Prepared {machine}/{run} with {len(records)} annotated intervals", flush=True)
    return checkpoint


def prepare(path, workers):
    discovered = defaultdict(dict)
    with ZipFile(path) as archive:
        for file in archive.namelist():
            names = file.split("/")
            if len(names) == 3 and names[-1]:
                kind = names[-1].rsplit("_", 1)[-1]
                if kind in {"label.csv", "acc.csv", "ct.csv", "mtc.csv", "s0.wav", "s1.wav", "s2.wav"}:
                    if kind in discovered[tuple(names[:2])]:
                        raise ValueError(f"Duplicate {kind} in {names[:2]}")
                    discovered[tuple(names[:2])][kind] = file
    stamp = path.stat()
    checkpoints = CACHE.parent / f"preprocessed-{stamp.st_size}-{stamp.st_mtime_ns}"
    checkpoints.mkdir(parents=True, exist_ok=True)
    jobs = [(path, machine, run, files, checkpoints)
            for (machine, run), files in sorted(discovered.items())]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        prepared = list(pool.map(lambda args: prepare_run(*args), jobs))
    rows = pd.concat([pd.read_csv(path) for path in prepared], ignore_index=True)
    rows.to_csv(CACHE.with_suffix(".tmp"), index=False)
    CACHE.with_suffix(".tmp").replace(CACHE)
    print(f"Saved {len(rows)} intervals from {len(discovered)} runs to {CACHE}", flush=True)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "dataset" / "MSM-Milling.zip")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    prepare(args.archive, args.workers)
