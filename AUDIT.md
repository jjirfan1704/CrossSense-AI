# CrossSense AI project audit

## Initial workspace

The repository contained a working fan research workflow represented by existing source and archived outputs. The copied virtual environment could not start on this machine. Only the fan data was present locally at the start of the audit. No bearing or CNC implementation, Streamlit entry point, or project setup guide was present.

| Existing file | Purpose and audit finding |
|---|---|
| src/preprocessing.py | Inventories WAV metadata and infers split from the folder and label from the filename. The saved inventory embedded absolute paths from another computer. |
| src/feature_extraction.py | Summarizes each clip with mean and standard deviation for 64 log-Mel bands. Assumes the saved inventory paths are accessible. |
| src/temporal_features.py | Averages log-Mel bands in ten one-second segments and flattens the results. The resulting dense vector preserves segment position but averages within-segment variation. |
| src/model.py | Fits a scaled dense reconstruction MLP. Calibrates its threshold on in-sample training errors. |
| src/model_temporal.py | Fits a dense reconstruction MLP on flattened segment features. Uses held-out normals for calibration. |
| src/model_isolation.py | Fits Isolation Forest on temporal features and negates scikit-learn's inlier score. Uses held-out normal calibration. |
| src/evaluation.py | Generates distributions, confusion matrices and ROC curves for the archived original Autoencoder only. |
| src/visualization.py | Plots waveform and spectrogram examples using the saved inventory. |
| src/detection.py | Empty integration stub. |
| main.py | Originally invoked Isolation Forest training from the temporal feature CSV. |
| requirements.txt | Listed the original scientific stack. Exact environment reproduction was limited by unpinned versions and the unusable copied interpreter. |

Original artifacts include three model/scaler/threshold sets, two feature tables, three prediction tables and four plots. The new workflows keep these as archived baselines.

## Fan findings

Local development filenames and folder names agree on the section, domain, label and split. For each section, there are 1,000 source and three target normal training WAVs, then 200 test WAVs per domain, balanced between normal and anomaly. Higher scores already meant more anomalous in all three archived implementations. The new extractor also checks every waveform for expected length, channel count and sample rate, and hashes the WAV bytes to detect exact train/test overlap.

There is no direct train/test leakage apparent in the legacy fitting code. Its biggest evaluation gaps were pooling sections, calibrating the first model in sample, and reporting aggregate domain scores without the section-level breakdown. Random source-normal calibration has a further limitation when the target normal distribution shifts. Near-random archive AUCs establish weak ranking. Section conditioning and retaining upper-tail spectral energy were the first revisions tested. Comparing old versus new results combines those changes and does not isolate their individual effect.

The source calibration threshold is frozen before test scoring. The saved standardized partial AUC through 10% FPR is a scikit-learn statistic. The official DCASE ranking uses its published protocol.

## Priorities and status

1. **Fan.** Preserve the archived experiments, implement portable WAV discovery and real inference, then evaluate fixed section-specific covariance and nearest-neighbor models with independent threshold calibration.
2. **Bearing.** Use an actual NASA IMS run, compare a covariance detector against Isolation Forest, and report chronological alarm behavior. Failure-window alarms are proxies because onset labels are absent.
3. **CNC.** Use actual expert-annotated cutting intervals, keep entire runs disjoint, compare separate power and vibration models and choose fusion weights using training-run folds. The verified full archive was processed after the controller pilot. The final benchmark includes all three machines, raw accelerometer data where available, and separate run-disjoint evaluations for each machine.
4. **Integration.** Use one dashboard to load saved models, run real recordings or intervals, and display model metrics, decision status and plots.

The reproducible train and launch commands are in [README.md](README.md). The presentation and demo notes are in [PRESENTATION.md](PRESENTATION.md).

## Sources

- [MIMII DUE dataset](https://zenodo.org/records/4740355)
- [Official DCASE 2021 evaluation protocol](https://dcase.community/challenge2021/task-unsupervised-detection-of-anomalous-sounds)
- [NASA IMS Bearings](https://data.nasa.gov/dataset/ims-bearings)
- [MSM Milling data descriptor](https://doi.org/10.1038/s41597-026-07255-7)
