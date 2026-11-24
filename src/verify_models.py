"""Check deployed scores against saved evaluation records and real inputs."""
import joblib
import numpy as np
import pandas as pd

from src import bearing, cnc, fan
from src.benchmark import DEFAULT_BEARING_RUN, ROOT, resolve_run
from src.cnc_prepare import CACHE


def check_fan():
    models = joblib.load(ROOT / "models/fan/benchmark.joblib")
    raw = {path.name: path for path, *_ in fan.recordings(ROOT / "dataset/fan")}
    table = pd.read_csv(ROOT / "results/fan/predictions.csv", dtype={"section": str})
    for (section, method), model in models.items():
        rows = table[(table.section == section) & (table.model == method)]
        for _, row in rows.groupby(["domain", "actual"]).head(1).iterrows():
            score, decision, *_ = fan.predict(raw[row.file], model)
            assert np.isclose(score, row.score), (section, method, score, row.score)
            assert decision == bool(row.predicted)
        print(f"Verified fan section {section} / {method}", flush=True)


def check_bearing():
    models = joblib.load(ROOT / "models/bearing/benchmark.joblib")
    table = pd.read_csv(ROOT / "results/bearing/predictions.csv")
    for method, model in models.items():
        rows = table[table.model == method]
        folder = resolve_run(model["run"], DEFAULT_BEARING_RUN)
        if folder is None:
            raise FileNotFoundError(f"Bearing snapshots are unavailable for {method}.")
        for row in (rows.iloc[0], rows.iloc[-1]):
            score, decision, *_ = bearing.predict(folder / row.file, model)
            assert np.isclose(score, row.score), (method, score, row.score)
            assert decision == bool(row.predicted)
        print(f"Verified bearing {method}", flush=True)


def check_cnc_full():
    filename = ROOT / "models/cnc_full/benchmark.joblib"
    if not filename.is_file():
        raise FileNotFoundError("Train the full CNC benchmark before running its smoke check.")
    models = joblib.load(filename)
    features = pd.read_csv(CACHE)
    table = pd.read_csv(ROOT / "results/cnc_full/predictions.csv", float_precision="round_trip")
    assert ((table.score > table.threshold).astype(int) == table.predicted).all()
    assert table.groupby(["machine", "run"]).fold.nunique().eq(1).all()
    for machine, pack in models.items():
        rows = features[features.machine == machine]
        for key, estimator in pack["estimators"].items():
            columns = pack["schemas"][key.split("/")[0]]
            records = rows.dropna(subset=columns).iloc[:2][columns].to_numpy()
            values = estimator.predict_proba(records)
            assert values.shape == (2, 2) and np.isfinite(values).all()
            assert np.allclose(values.sum(axis=1), 1)
        assert np.isclose(sum(pack["weights"].values()), 1)
        from src.cnc_full import predict_features
        for _, row in rows.head(2).iterrows():
            value, _, scores = predict_features(row, pack, "validation_fusion")
            assert 0 <= value <= 1 and len(scores) == len(pack["weights"])
        print(f"Verified {machine}: {len(pack['estimators'])} CNC estimators", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-full", action="store_true")
    options = parser.parse_args()
    check_fan()
    check_bearing()
    if not options.skip_full:
        check_cnc_full()
