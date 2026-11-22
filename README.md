# CrossSense AI

CrossSense AI monitors industrial fan audio, bearing vibration and CNC milling sensors. The Streamlit app loads trained detectors and shows their scores, classifications and evaluation outputs.

## Setup on Windows

Use Python 3.12. Create a fresh environment for this checkout.

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## Working from a GitHub clone

The repository includes trained model bundles, evaluation outputs, and the dashboard. Install the dependencies in a fresh virtual environment. Raw recordings live under dataset/ and need to be obtained separately from the sources below. Fan and bearing file uploads work with the trained models. Local recording pickers and retraining require the corresponding data under dataset/. The full CNC demo can use its included interval feature table.

For the archived fan scripts, regenerate the machine-local inventory with python -m src.preprocessing before re-extracting features.
## Train, evaluate and launch

Each training command reads raw files, extracts features, fits detectors, evaluates held-out recordings and writes models, CSVs and figures. The current workspace also includes an MSM controller-only experiment and its held-out metrics.

```powershell
.venv/Scripts/python.exe main.py fan --data dataset/fan
.venv/Scripts/python.exe -m src.legacy_metrics
.venv/Scripts/python.exe main.py bearing --data dataset/bearing/IMS/2nd_test --channel 1 --rate 20000
.venv/Scripts/python.exe main.py cnc --data dataset/cnc-controller/imi_vm20i
.venv/Scripts/python.exe -m streamlit run dashboard.py
```

The NASA IMS ZIP contains nested RAR archives. Extract the second test into the bearing dataset path above. Choose the channel and sampling rate using the archive documentation. CNC expects one machine folder containing separate run directories. The project stores new outputs in models/<machine>, results/<machine> and graphs/<machine>. Saved Joblib files are executable Python serialization. Load local artifacts created by these training commands.

## Fan

The development subset contains sections 00, 01 and 02. Each section has 1,000 source-normal and three target-normal training WAVs, plus 100 normal and 100 anomalous WAVs per test domain. The loader checks the filename against its folder, verifies waveform format and rejects byte-identical train/test overlap.

For each section, extract log-spaced STFT band means, standard deviations and 90th percentiles. Compare 10-neighbor distance and shrinkage Mahalanobis distance. Fit on 80% of source normals and the three provided target normals. Set the threshold at the 95th percentile of the remaining source-normal scores. Evaluate the fixed models on source and target test recordings. The target false-positive rate can shift because the calibration distribution is source-only.

The reported partial AUC is scikit-learn's standardized partial AUC through 10% FPR. DCASE publishes its official ranking procedure separately. These development-set results measure performance within the available sections.

## Bearing

Train on the first 20% of a single run and calibrate on the next 10%. Use chronological snapshots and compare covariance distance with Isolation Forest. Features include RMS, variance, kurtosis and spectral summaries. The dashboard shows the score trend and flags three consecutive threshold crossings. The final 10% is a late-life proxy for a channel known to have failed. Exact degradation onset is unlabeled, so the benchmark reports alarm rates and proxy detection rates without ROC or classification accuracy.

## CNC milling

Use synchronized, numeric accelerometer, current and selected MTConnect vibration/power CSV channels when their schemas match across runs. Extract statistics over the expert-annotated time intervals. Use label 0 for stable cutting and label 1 for abnormal cutting. The separate tool-defect label remains outside this binary experiment. Entire runs are held out. Sensor-specific Random Forests produce anomaly scores on the same intervals. On the downloaded controller data for imi_vm20i, one model uses power and another uses the four RMS vibration channels. Both derive from the same time-aligned MTConnect CSV. The demo model has been evaluated on held-out runs within this machine. When two sensors are available, blend their outputs with weights selected on training-run cross-validation. If the training folds cannot support weight selection, the blend uses equal weights. The dashboard expects CSVs from the trained configuration and the start and end of a cutting interval.

## Revised fan and bearing models

```powershell
.venv/Scripts/python.exe -m src.fan_supervised
.venv/Scripts/python.exe -m src.bearing_reference
```

The supervised spectral Extra Trees model is available on the fan page. The new experiment repurposes 140 of the 200 labeled source development clips in each section for fitting, then evaluates on the other 60 source clips and all 200 target clips. It also uses 250 original source-normal training clips per section. The target alignment candidate uses the three provided target-normal training clips. Target test recordings are held out of fitting and threshold selection. This is a supervised development experiment. The dashboard still offers the original normal-only benchmark.

Extra Trees averaged 82.8% source accuracy and 61.3% target accuracy, with recall of 65.6% and 57.3%. The original covariance scores on the same files averaged 51.1% and 51.3% accuracy, with recall of 7.8% and 13.0%. SVM and target-aligned Extra Trees are included in the comparison CSV.

The bearing page defaults to the 100–1,000 Hz vibration energy detector with chronological confirmation when local snapshot history is available. Thresholds for the new candidates use the 99th percentile of the original early chronological calibration window. The low-band detector flagged 97 of the final 99 proxy snapshots and 40.2% of all 689 scored snapshots. The original covariance detector flagged 99 of 99 and 68.9% overall. Both operate on real vibration measurements. The late proxy and overall alarm rate cannot establish classification accuracy or a false-positive rate.

## Segment classifiers and alarm policies

```powershell
.venv/Scripts/python.exe -m src.fan_segments
.venv/Scripts/python.exe -m src.alarm_policies
```

The default fan classifier averages SVM scores from five two-second segments. Gain-normalized spectral features reduce reliance on overall loudness. On the same development partitions, segment SVM reached 87.8% source accuracy and 75.6% source recall, and 62.3% target accuracy and 68.0% target recall. The previous Extra Trees model remains available and has higher target ROC-AUC.

The bearing alarm uses three positive votes in the latest five chronological snapshots. On the same IMS replay it flagged 99 of 99 late-life proxy snapshots and 39.0% of the scored stream. With a single uploaded snapshot, the dashboard only has enough information for a single-snapshot decision.

For industrial CNC, the dashboard selects controller power if available and microphone S0 otherwise. Replaying the saved run-held-out predictions gave 90.5% accuracy, 85.7% recall and 66.7% F1. The replay caught six of seven abnormal intervals, versus four with validation-weighted fusion, and produced two more false alarms. The fallback was designed after examining the earlier results, so confirm it on additional industrial runs.

## Full CNC benchmark

The complete archive is stored at dataset/MSM-Milling.zip. Its size is 14,203,163,495 bytes, and all 481 ZIP entries passed CRC verification. Preprocessing reads the ZIP in place and checkpoints each run. Use fewer preprocessing workers on a machine with limited memory.

```powershell
.venv/Scripts/python.exe -m src.cnc_prepare --workers 4
.venv/Scripts/python.exe -m src.cnc_full
.venv/Scripts/python.exe -m src.verify_models
```

The full benchmark uses the archive's available accelerometer, current, microphone and MTConnect power, load or vibration signals. It summarizes each expert cutting interval, compares Random Forest and Extra Trees for each sensor, and evaluates equal and validation-weighted fusion. Each machine is evaluated separately because sensor configurations differ. Each model is scored on intervals where its sensor is available. Check the test sample counts before comparing sensor metrics. The fusion score combines the available sensors and falls back to their mean when the preferred sensor is missing. Tool-defect annotations are preserved in the feature table and excluded from the stable-versus-abnormal cutting benchmark.

Entire machining runs stay together in the outer test folds. Inner validation selects model and fusion weights and targets at least 90% recall while minimizing the validation false-positive rate. Held-out recall can differ, especially on the industrial machine with few abnormal intervals. The dashboard's full archive demo uses the saved models fitted on all eligible intervals. Its evaluation panels use the held-out predictions.

See results/cnc_full/coverage.csv for eligible runs and intervals, metrics.csv for test performance, matched_metrics.csv for comparisons on shared intervals, and predictions.csv for fold-level scores and decisions.
## Existing experiments

The original fan scripts and artifacts remain in src, models, results and graphs. The old dataset inventory contains absolute paths from a previous machine. The legacy autoencoder calibrates on its training errors. The temporal model flattens one-second Mel summaries into a dense network, and both it and the legacy Isolation Forest pool the three sections. The original score directions are consistent with higher scores indicating more anomalous recordings. AUC around 0.5 across several baselines indicates poor ranking. Threshold changes affect the operating point, while feature and modeling changes are needed to improve ranking.

## Evaluation snapshot

| Module | Audited result |
|---|---|
| Fan | Covariance mean section ROC-AUC was about 0.64 source and 0.63 target. Mean F1 was about 0.19 and 0.20 at the fixed threshold. |
| Bearing | Both models flagged all 99 late-life proxy snapshots along with about 69% of the full scored stream. Exact onset is unknown. |
| CNC full archive | Representative held-out results were 95.5% accuracy / 92.0% recall for imi_vm20i microphone S1, 95.8% / 91.7% for imi_vmx30ui microphone S2, and 94.9% / 83.3% for tmf_vf10 controller power on its 59 covered intervals. Full-coverage validation fusion on tmf_vf10 detected four of seven anomalies. |

See [AUDIT.md](AUDIT.md) for the original pipeline review and [PRESENTATION.md](PRESENTATION.md) for the review slides and notes.

## Review demo

1. Open the overview and check each module's model status.
2. Select a fan section and compare a labeled normal and anomalous development WAV, then show source and target evaluation side by side.
3. Select a bearing snapshot and explain the channel features, chronological split and persistent alarm.
4. Open the CNC full archive view and select a machine, run and cutting interval. Compare the saved detector scores with held-out metrics. The controller CSV upload mode supports the original controller model.
5. Explain ROC-AUC, the calibration threshold and the practical trade-off between missed detections and false alarms.

## Dataset sources

- [MIMII DUE on Zenodo](https://zenodo.org/records/4740355)
- [DCASE 2021 Task 2 evaluation](https://dcase.community/challenge2021/task-unsupervised-detection-of-anomalous-sounds)
- [NASA IMS Bearings](https://data.nasa.gov/dataset/ims-bearings)
- [MSM Milling dataset on Kaggle](https://www.kaggle.com/datasets/manufuturetoday/multi-sensor-for-metal-milling-anomaly)
- [MSM Milling data descriptor](https://doi.org/10.1038/s41597-026-07255-7)






