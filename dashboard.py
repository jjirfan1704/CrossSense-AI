"""Launch with python -m streamlit run dashboard.py."""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import bearing, bearing_reference, cnc, fan, fan_segments, fan_supervised
from src.benchmark import DEFAULT_BEARING_RUN, ROOT, resolve_run
from src.alarm_policies import bearing_alarm
from src.plant_health import (
    RECOMMENDATIONS,
    build_maintenance_report_md,
    compute_plant_summary,
    get_replay_series,
    send_slack_alert,
)

MACHINES = {"Industrial Fan": "fan", "Industrial Bearing": "bearing",
            "CNC Milling Machine": "cnc"}

st.set_page_config(page_title="CrossSense AI", page_icon="📈", layout="wide")
st.title("CrossSense AI")
page = st.sidebar.radio("Machine", ["Overview", *MACHINES])


@st.cache_resource
def load_models(path, modified):
    return joblib.load(path)


def performance(machine, machine_id=None):
    folder = ROOT / "results" / machine
    if not (folder / "metrics.csv").is_file():
        return
    st.subheader("Evaluation")
    metrics = pd.read_csv(folder / "metrics.csv")
    if machine_id:
        metrics = metrics[metrics.machine.eq(machine_id)]
    st.dataframe(metrics)
    if machine in ("fan", "fan_supervised", "fan_segments"):
        legacy_path = ROOT / "results" / "fan_legacy_metrics.csv"
        if machine == "fan_supervised":
            legacy_path = folder / "matched_baseline.csv"
        if machine == "fan_segments":
            legacy_path = ROOT / "results/fan_supervised/metrics.csv"
        if legacy_path.is_file():
            legacy = pd.read_csv(legacy_path)
            comparison = pd.concat([metrics, legacy])
            columns = ["accuracy", "recall", "f1", "roc_auc"]
            st.caption("Mean across sections for each test domain")
            st.dataframe(comparison.groupby(["model", "domain"])[columns].mean())
    if machine == "bearing_reference":
        persistence = folder / "persistence_metrics.csv"
        if persistence.is_file():
            st.caption("Chronological low-band alarm rules")
            st.dataframe(pd.read_csv(persistence))
        baseline = ROOT / "results" / "bearing" / "metrics.csv"
        if baseline.is_file():
            st.caption("Original models on the same chronological snapshots")
            st.dataframe(pd.read_csv(baseline))
    matrix_path = ROOT / "graphs" / machine / "confusion_matrices.png"
    if matrix_path.is_file() and not machine_id:
        with st.expander("Confusion matrices"):
            st.image(str(matrix_path))
    curves = folder / "roc.csv"
    if curves.is_file():
        roc = pd.read_csv(curves)
        if machine_id:
            roc = roc[roc.machine.eq(machine_id)]
            selected = st.selectbox("ROC detector", roc.model.unique())
            roc = roc[roc.model.eq(selected)]
        fig, ax = plt.subplots()
        groups = [name for name in ("model", "section", "domain") if name in roc]
        for key, frame in roc.groupby(groups):
            ax.plot(frame.fpr, frame.tpr, label=" / ".join(map(str, key)))
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set(xlabel="False positive rate", ylabel="True positive rate")
        ax.legend(fontsize="small", ncol=2)
        st.pyplot(fig)
        plt.close(fig)
    scores = pd.read_csv(folder / "predictions.csv")
    if machine_id:
        scores = scores[scores.machine.eq(machine_id)]
    model = st.selectbox("Evaluation score series", scores.model.unique())
    sample = scores[scores.model.eq(model)]
    columns = ["score", "threshold"]
    if machine.startswith("bearing"):
        st.caption("Late-life proxy assumes the selected channel is from the failed bearing. Onset labels are unavailable.")
        sample = sample.set_index("timestamp")
        columns.append("rms")
    if not machine.startswith("bearing"):
        sample = sample.reset_index(drop=True)
    st.line_chart(sample[columns])


def status(value, anomalous):
    st.metric("Anomaly score", f"{value:.4g}")
    (st.warning if anomalous else st.success)("Anomaly" if anomalous else "Normal")


if page == "Overview":
    st.caption("Unified industrial condition monitoring for acoustic (MIMII DUE), vibration (NASA IMS), and CNC milling (MSM) sensors.")
    summary = compute_plant_summary()
    overall_health = round(sum(item["health_pct"] for item in summary) / len(summary), 1)
    active_alerts = sum(1 for item in summary if item["status"] != "Healthy")
    total_samples = sum(item["samples"] for item in summary)

    k1, k2, k3 = st.columns(3)
    k1.metric("Overall Plant Health", f"{overall_health}%")
    k2.metric("Active Machine Alerts", f"{active_alerts} / {len(summary)}")
    k3.metric("Evaluated Windows", f"{total_samples:,}")

    st.subheader("Machine Health & Predictive Maintenance")
    for panel, item in zip(st.columns(3), summary):
        panel.subheader(item["machine"])
        panel.metric("Health Index", f"{item['health_pct']}%", delta=item["status"],
                     delta_color="normal" if item["status"] == "Healthy" else "inverse")
        panel.write(f"**Latest score:** `{item['latest_score']}` (threshold `{item['threshold']}`)")
        panel.write(f"**Recent alarm rate:** `{item['recent_alarm_rate']}%`")
        (panel.success if item["status"] == "Healthy" else (panel.warning if item["status"] == "Warning" else panel.error))(
            item["recommendation"]
        )

    st.subheader("Live Replay Simulator & Slack Alerts")
    sim_col, slack_col = st.columns([2, 1])
    with sim_col:
        replay_machine = st.selectbox("Simulate live stream for", ["fan", "bearing", "cnc"],
                                      format_func=lambda k: {"fan": "Industrial Fan", "bearing": "Industrial Bearing", "cnc": "CNC Milling"}[k])
        replay_df = get_replay_series(replay_machine, limit=120)
        if not replay_df.empty:
            step = st.slider("Replay step (chronological window)", min_value=5, max_value=len(replay_df), value=min(40, len(replay_df)))
            sliced = replay_df.iloc[:step].set_index("step")
            st.line_chart(sliced[["score", "threshold"]])
            anoms = int(sliced["predicted"].sum())
            st.caption(f"Replayed {step} windows — {anoms} anomalies flagged ({anoms / step * 100:.1f}%).")
    with slack_col:
        st.markdown("**Shift Reports & Slack Dispatch**")
        report_md = build_maintenance_report_md(summary)
        report_csv = pd.DataFrame(summary).to_csv(index=False)
        st.download_button("Download Shift Report (Markdown)", data=report_md, file_name="crosssense_shift_report.md", mime="text/markdown")
        st.download_button("Download Shift Report (CSV)", data=report_csv, file_name="crosssense_shift_report.csv", mime="text/csv")
        slack_channel = st.text_input("Slack Channel ID", value="C0CL0J5KF5X")
        slack_token = st.text_input("Slack Bot Token (optional if env set)", type="password")
        if st.button("Send Alert to Slack"):
            ok, msg = send_slack_alert(summary, token=slack_token or None, channel=slack_channel)
            (st.success if ok else st.error)(msg)
    st.stop()

machine = MACHINES[page]
st.header(page)
full_cnc = ROOT / "models" / "cnc_full" / "benchmark.joblib"
if machine == "cnc" and full_cnc.is_file():
    version = st.selectbox("CNC model", ["Full archive", "Controller CSV uploads"])
    if version == "Full archive":
        from src.cnc_full import predict_features
        from src.cnc_prepare import CACHE

        packs = load_models(full_cnc, full_cnc.stat().st_mtime_ns)
        machine_id = st.selectbox("Machine configuration", list(packs))
        pack = packs[machine_id]
        options = ["validation_fusion", *pack["estimators"]]
        power = "mtc_power/random_forest"
        microphone = "s0/random_forest"
        fallback = machine_id == "tmf_vf10" and {power, microphone} <= set(pack["estimators"])
        if fallback:
            options.insert(0, "power_then_s0")
        method = st.selectbox("Detector", options)
        features = pd.read_csv(CACHE)
        features = features[features.machine.eq(machine_id) & features.actual.isin([0, 1])]
        sensors = ([key.split("/")[0] for key in pack["weights"]] if method == "validation_fusion"
                   else [method.split("/")[0]])
        if method == "power_then_s0":
            sensors = ["mtc_power", "s0"]
        usable = np.logical_or.reduce([features[pack["schemas"][sensor]].notna().all(axis=1)
                                       for sensor in sensors])
        features = features[usable]
        run = st.selectbox("Archive run", sorted(features.run.unique()))
        intervals = features[features.run.eq(run)]
        index = st.selectbox("Cutting interval", list(intervals.index),
                             format_func=lambda i: f"{intervals.at[i, 'start']:g}–{intervals.at[i, 'end']:g} s")
        row = intervals.loc[index]
        detector = method
        if method == "power_then_s0":
            detector = power if row[pack["schemas"]["mtc_power"]].notna().all() else microphone
            st.caption(f"Sensor used for this interval: {detector}")
        value, anomalous, scores = predict_features(row, pack, detector)
        status(value, anomalous)
        st.dataframe(pd.Series(scores, name="Sensor score").to_frame())
        if len(scores) > 1:
            st.markdown("**Per-sensor anomaly contribution**")
            st.bar_chart(pd.Series(scores, name="Score"))
        (st.warning if anomalous else st.info)(RECOMMENDATIONS["cnc"]["Critical" if anomalous else "Healthy"])
        st.caption("This example uses the model fitted on all eligible runs. Evaluation below uses held-out runs.")
        if method == "power_then_s0":
            policy_metrics = ROOT / "results/cnc_full/industrial_fallback_metrics.csv"
            if policy_metrics.is_file():
                st.dataframe(pd.read_csv(policy_metrics))
        performance("cnc_full", machine_id)
        st.stop()
benchmark = machine
available_families = {
    "fan": [("fan_segments", "Segment classifiers"), ("fan_supervised", "Spectral supervised"),
            ("fan", "Normal-only baseline")],
    "bearing": [("bearing_reference", "Vibration band and reference"), ("bearing", "Original models")],
}
if machine in available_families:
    families = {label: key for key, label in available_families[machine]
                if (ROOT / "models" / key / "benchmark.joblib").is_file()}
    if families:
        benchmark = families[st.selectbox("Model family", list(families))]
artifact = ROOT / "models" / benchmark / "benchmark.joblib"
if not artifact.is_file():
    st.info(f"The {machine} module is awaiting dataset training. See README.md for setup.")
    st.stop()

try:
    model_pack = load_models(artifact, artifact.stat().st_mtime_ns)
    if machine == "fan":
        section = st.selectbox("Recording section", sorted({key[0] for key in model_pack}))
        methods = [name for sec, name in model_pack if sec == section]
        if benchmark == "fan_supervised":
            methods.sort(key=lambda name: (name != "extra_trees", name))
            st.caption("Trained on labeled source development clips. Examples below are held out.")
        method = st.selectbox("Detector", methods)
        wav = st.file_uploader("Upload mono 16 kHz, 10-second WAV", type="wav")
        examples = [path for path, sec, _, split, _ in fan.recordings(ROOT / "dataset" / "fan")
                    if sec == section and split == "test"]
        if benchmark in ("fan_supervised", "fan_segments"):
            tested = set(pd.read_csv(ROOT / "results" / benchmark / "predictions.csv").file)
            examples = [path for path in examples if path.name in tested]
        sample = st.selectbox("Or choose a development recording", examples,
                              index=None, format_func=lambda path: path.name)
        source = wav if wav is not None else sample
        if source is not None:
            if benchmark == "fan_segments":
                value, anomalous, signal, times, spectral = fan_segments.predict(
                    source, model_pack[section, method])
            elif benchmark == "fan_supervised":
                domain = (sample.name.split("_")[2] if wav is None
                          else st.selectbox("Recording domain", ["source", "target"]))
                value, anomalous, signal, times, spectral = fan_supervised.predict(
                    source, model_pack[section, method], domain)
            else:
                value, anomalous, signal, times, spectral = fan.predict(source, model_pack[section, method])
            status(value, anomalous)
            st.audio(wav.getvalue() if wav is not None else sample.read_bytes(), format="audio/wav")
            st.line_chart(pd.Series(signal[::20], index=np.arange(0, len(signal), 20) / 16000,
                                    name="Amplitude"), x_label="Seconds")
            fig, ax = plt.subplots()
            ax.pcolormesh(times, np.arange(len(spectral)), spectral, shading="auto")
            ax.set(xlabel="Seconds", ylabel="Log-spaced frequency band")
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("**Mean spectral energy per frequency band**")
            st.bar_chart(pd.Series(np.mean(spectral, axis=1), name="Band energy"))
            (st.warning if anomalous else st.info)(RECOMMENDATIONS["fan"]["Critical" if anomalous else "Healthy"])
    elif machine == "bearing":
        methods = list(model_pack)
        if benchmark == "bearing_reference":
            methods.sort(key=lambda name: (name != "low_band", name))
        method = st.selectbox("Detector", methods)
        model = model_pack[method]
        st.caption(f"Channel {model['channel']} at {model['rate']:g} Hz")
        uploaded = st.file_uploader("Upload an IMS vibration snapshot")
        folder = resolve_run(model["run"], DEFAULT_BEARING_RUN)
        if folder is None:
            st.caption(f"Snapshot folder unavailable. Place the IMS run in {DEFAULT_BEARING_RUN} or upload a snapshot.")
        examples = sorted([p for p in folder.glob("*") if p.is_file() and p.name.count(".") == 5]) if folder else []
        sample = st.selectbox("Or choose an IMS snapshot", examples,
                              index=None, format_func=lambda path: path.name)
        source = uploaded if uploaded is not None else sample
        if source is not None:
            predictor = bearing_reference.predict if benchmark == "bearing_reference" else bearing.predict
            value, anomalous, features, signal = predictor(source, model)
            if benchmark == "bearing_reference" and method == "low_band" and uploaded is None:
                index = examples.index(sample)
                if index >= 4:
                    history = [predictor(path, model)[0] for path in examples[index - 4:index]]
                    anomalous = bool(bearing_alarm([*history, value], model["threshold"], 5, 3)[-1])
                    st.caption("Alarm uses at least three threshold crossings in the last five snapshots.")
                else:
                    st.caption("Fewer than five snapshots are available. Showing the single-snapshot result.")
            status(value, anomalous)
            stride = max(1, len(signal) // 3000)
            st.line_chart(pd.Series(signal[::stride], index=np.arange(0, len(signal), stride) / model["rate"],
                                    name="Vibration"), x_label="Seconds")
            st.dataframe(pd.Series(features, name="Value").to_frame())
    else:
        st.caption(f"Sensor configuration for {model_pack['machine']}")
        source_names = sorted({cnc.source_modality(name) for name in model_pack["schemas"]})
        files = {name: st.file_uploader(f"Upload {name} sensor CSV", type="csv", key=name)
                 for name in source_names}
        source_folder = model_pack.get("data_root")
        default_start, default_end = 0.0, 1.0
        sensors = {}
        if all(files.values()):
            sensors = {name: cnc.read_sensor(file, name) for name, file in files.items()}
        elif source_folder is not None and not any(files.values()):
            data_root = ROOT / source_folder
            runs = sorted((data_root).rglob("label.csv"))
            run = st.selectbox("Or choose a local cutting run", runs, index=None,
                               format_func=lambda path: path.parent.name)
            if run is not None:
                labels = pd.read_csv(run)
                labels.columns = labels.columns.str.strip().str.lower()
                intervals = list(labels[["start", "end"]].itertuples(index=False, name=None))
                default_start, default_end = st.selectbox(
                    "Annotated cutting interval", intervals,
                    format_func=lambda item: f"{item[0]:g}–{item[1]:g} seconds")
                for name in files:
                    matches = list(run.parent.glob(f"*_{name}.csv"))
                    if len(matches) != 1:
                        raise ValueError(f"The selected run needs one {name} CSV.")
                    sensors[name] = cnc.read_sensor(matches[0], name)
        start = st.number_input("Cutting interval start (seconds)", value=float(default_start),
                                key=f"start-{default_start}-{default_end}")
        end = st.number_input("Cutting interval end (seconds)", value=float(default_end),
                              key=f"end-{default_start}-{default_end}")
        if sensors:
            scores, value, anomalous = cnc.predict(sensors, start, end, model_pack)
            status(value, anomalous)
            if len(scores) > 1:
                st.caption("The anomaly score is the weighted blend of sensor scores.")
            st.dataframe(pd.DataFrame({"Sensor anomaly score": scores,
                                       "Fusion weight": model_pack["weights"]}))
            for name, frame in sensors.items():
                channels = list(dict.fromkeys(column for view, columns in model_pack["schemas"].items()
                                              if cnc.source_modality(view) == name for column in columns))
                frame = frame.loc[start:end, channels]
                st.markdown(f"**{name} signals**")
                st.line_chart(frame.iloc[::max(1, len(frame) // 3000)])
except (ValueError, OSError, RuntimeError, KeyError) as error:
    st.error(f"Could not score the recording. {error}")

performance(benchmark)
