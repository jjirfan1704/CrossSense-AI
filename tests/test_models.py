"""Dataset-free checks for the trained CrossSense AI model bundles."""
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src import bearing, bearing_reference, fan, fan_segments, fan_supervised
from src.benchmark import DEFAULT_BEARING_RUN, ROOT, portable_path, resolve_run
from src.cnc_full import predict_features

MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def noisy_wav(path, seed):
    rate = 16000
    signal = 0.01 * np.random.default_rng(seed).standard_normal(10 * rate).astype("float32")
    sf.write(path, signal, rate, subtype="FLOAT")


def noisy_snapshot(path, seed):
    data = 0.05 * np.random.default_rng(seed).standard_normal((40960, 4))
    np.savetxt(path, data)


def test_fan_bundles_score_a_recording(tmp_path):
    packs = {"fan": fan, "fan_segments": fan_segments, "fan_supervised": fan_supervised}
    for name, module in packs.items():
        pack = joblib.load(MODELS / name / "benchmark.joblib")
        section = sorted({key[0] for key in pack})[0]
        method = sorted(name_ for sec, name_ in pack if sec == section)[0]
        wav = tmp_path / f"{name}.wav"
        noisy_wav(wav, seed=1)
        value, _, *_ = module.predict(wav, pack[section, method])
        assert np.isfinite(value)


def test_bearing_bundles_score_a_snapshot(tmp_path):
    for name, module in (("bearing", bearing), ("bearing_reference", bearing_reference)):
        pack = joblib.load(MODELS / name / "benchmark.joblib")
        model = pack[sorted(pack)[0]]
        snapshot = tmp_path / f"{name}.txt"
        noisy_snapshot(snapshot, seed=2)
        value, _, *_ = module.predict(snapshot, model)
        assert np.isfinite(value)


@pytest.mark.parametrize("machine", ["imi_vm20i", "imi_vmx30ui", "tmf_vf10"])
def test_cnc_full_bundle_scores_each_machine(machine):
    pack = joblib.load(MODELS / "cnc_full" / "benchmark.joblib")[machine]
    features = pd.read_csv(RESULTS / "cnc_full" / "features.csv")
    features = features[features.machine.eq(machine) & features.actual.isin([0, 1])]
    sensors = [key.split("/")[0] for key in pack["weights"]]
    usable = np.logical_or.reduce([features[pack["schemas"][sensor]].notna().all(axis=1)
                                   for sensor in sensors])
    row = features[usable].iloc[0]
    value, anomalous, scores = predict_features(row, pack, "validation_fusion")
    assert 0 <= value <= 1 and isinstance(anomalous, bool) and scores


def test_saved_predictions_match_thresholds():
    for name in ("fan", "bearing", "cnc_full"):
        frame = pd.read_csv(RESULTS / name / "predictions.csv", float_precision="round_trip")
        predicted = (frame.score > frame.threshold).astype(int)
        assert predicted.eq(frame.predicted.astype(int)).all(), name


def test_portable_path_and_run_resolution(tmp_path, monkeypatch):
    import src.benchmark

    monkeypatch.setattr(src.benchmark, "ROOT", tmp_path)
    inside = tmp_path / "dataset" / "bearing" / "IMS" / "2nd_test"
    inside.mkdir(parents=True)
    assert portable_path(inside) == str(Path("dataset/bearing/IMS/2nd_test"))
    assert portable_path(tmp_path / "elsewhere").endswith("elsewhere")
    assert resolve_run("dataset/bearing/IMS/2nd_test") == inside
    assert resolve_run("H:/missing/run", DEFAULT_BEARING_RUN) == inside
    assert resolve_run("H:/missing/run") is None
