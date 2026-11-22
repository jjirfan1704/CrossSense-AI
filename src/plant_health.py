"""Plant-wide health scoring, predictive maintenance recommendations, report export, and Slack alerts."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.benchmark import ROOT

RECOMMENDATIONS = {
    "fan": {
        "Healthy": "Acoustic profile nominal. Continue standard 16 kHz periodic monitoring.",
        "Warning": "Elevated spectral energy detected in fan audio. Schedule intake/blade inspection and check for imbalance.",
        "Critical": "Persistent acoustic anomaly detected. Inspect fan bearings, motor mountings, and airflow obstruction immediately.",
    },
    "bearing": {
        "Healthy": "Vibration energy within early-life baseline. Maintain normal lubrication schedule.",
        "Warning": "Rising 100-1,000 Hz vibration energy. Plan bearing lubrication check and vibration re-test.",
        "Critical": "Sustained 3-of-5 bearing threshold crossings. High risk of inner/outer race degradation; schedule bearing replacement.",
    },
    "cnc": {
        "Healthy": "Spindle power and acoustic/vibration sensors nominal across cutting intervals.",
        "Warning": "Intermittent multisensor deviation during cutting. Inspect tool wear and coolant/feed parameters.",
        "Critical": "Abnormal cutting condition confirmed by multisensor fusion. Halt run to inspect tool integrity and spindle load.",
    },
}


def _load_latest(candidates: list[Path]) -> pd.DataFrame | None:
    for path in candidates:
        if path.is_file():
            return pd.read_csv(path)
    return None


def _compute_status(health_pct: float, recent_alarm_rate: float) -> str:
    if health_pct < 60.0 or recent_alarm_rate >= 0.4:
        return "Critical"
    if health_pct < 85.0 or recent_alarm_rate >= 0.15:
        return "Warning"
    return "Healthy"


def compute_plant_summary(window: int = 25) -> list[dict[str, Any]]:
    """Compute normalized 0-100% health, status, and maintenance advice for Fan, Bearing, and CNC."""
    specs = [
        ("fan", "Industrial Fan (MIMII DUE)", [ROOT / "results/fan_segments/predictions.csv", ROOT / "results/fan/predictions.csv"]),
        ("bearing", "Industrial Bearing (NASA IMS)", [ROOT / "results/bearing_reference/predictions.csv", ROOT / "results/bearing/predictions.csv"]),
        ("cnc", "CNC Milling (MSM Archive)", [ROOT / "results/cnc_full/predictions.csv", ROOT / "results/cnc/predictions.csv"]),
    ]
    rows: list[dict[str, Any]] = []
    for key, label, paths in specs:
        df = _load_latest(paths)
        if df is None or df.empty:
            rows.append({
                "key": key,
                "machine": label,
                "health_pct": 100.0,
                "status": "Healthy",
                "recent_alarm_rate": 0.0,
                "latest_score": 0.0,
                "threshold": 1.0,
                "samples": 0,
                "recommendation": RECOMMENDATIONS[key]["Healthy"],
            })
            continue

        tail = df.tail(window)
        alarm_rate = float(tail["predicted"].astype(float).mean()) if "predicted" in tail else 0.0
        latest = tail.iloc[-1]
        score = float(latest["score"])
        thresh = float(latest["threshold"]) if "threshold" in latest and float(latest["threshold"]) > 0 else 1.0
        ratio = max(0.0, score / thresh)
        health = float(np.clip(100.0 * (1.0 - 0.55 * alarm_rate - 0.25 * max(0.0, ratio - 0.8)), 5.0, 100.0))
        status_label = _compute_status(health, alarm_rate)
        rows.append({
            "key": key,
            "machine": label,
            "health_pct": round(health, 1),
            "status": status_label,
            "recent_alarm_rate": round(alarm_rate * 100.0, 1),
            "latest_score": round(score, 4),
            "threshold": round(thresh, 4),
            "samples": int(len(df)),
            "recommendation": RECOMMENDATIONS[key][status_label],
        })
    return rows


def get_replay_series(machine_key: str, limit: int = 120) -> pd.DataFrame:
    """Return a chronological slice of predictions for live replay simulation."""
    mapping = {
        "fan": [ROOT / "results/fan_segments/predictions.csv", ROOT / "results/fan/predictions.csv"],
        "bearing": [ROOT / "results/bearing_reference/predictions.csv", ROOT / "results/bearing/predictions.csv"],
        "cnc": [ROOT / "results/cnc_full/predictions.csv", ROOT / "results/cnc/predictions.csv"],
    }
    df = _load_latest(mapping[machine_key])
    if df is None or df.empty:
        return pd.DataFrame(columns=["step", "score", "threshold", "predicted"])
    sub = df.tail(limit).reset_index(drop=True).copy()
    sub["step"] = np.arange(1, len(sub) + 1)
    return sub[["step", "score", "threshold", "predicted"]]


def build_maintenance_report_md(summary: list[dict[str, Any]]) -> str:
    """Generate a Markdown predictive maintenance shift report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# CrossSense AI — Predictive Maintenance Shift Report",
        f"Generated: {now}",
        "",
        "| Machine | Health | Status | Recent Alarm Rate | Latest Score | Threshold | Recommended Action |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in summary:
        lines.append(
            f"| {item['machine']} | {item['health_pct']}% | {item['status']} | "
            f"{item['recent_alarm_rate']}% | {item['latest_score']} | {item['threshold']} | {item['recommendation']} |"
        )
    return "\n".join(lines) + "\n"


def send_slack_alert(summary: list[dict[str, Any]], token: str | None = None, channel: str = "C0CL0J5KF5X") -> tuple[bool, str]:
    """Send a formatted plant health & anomaly alert to Slack."""
    bot_token = (token or os.environ.get("SLACK_BOT_TOKEN") or os.environ.get("DSH_SUPERAGENT_SLACK_TOKEN") or "").strip()
    if not bot_token:
        return False, "No Slack bot token provided (set SLACK_BOT_TOKEN or paste token in sidebar)."

    alerts = [item for item in summary if item["status"] != "Healthy"]
    header = "🚨 *CrossSense AI — Predictive Maintenance Alert*" if alerts else "✅ *CrossSense AI — Plant Health Nominal*"
    lines = [header]
    for item in summary:
        icon = "🔴" if item["status"] == "Critical" else ("🟡" if item["status"] == "Warning" else "🟢")
        lines.append(
            f"{icon} *{item['machine']}*: {item['health_pct']}% Health ({item['status']}) "
            f"| Score `{item['latest_score']}` / Threshold `{item['threshold']}`\n"
            f"   _Action:_ {item['recommendation']}"
        )

    resp = requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={"Authorization": f"Bearer {bot_token}", "Content-Type": "application/json; charset=utf-8"},
        json={"channel": channel, "text": "\n".join(lines)},
        timeout=10,
    )
    data = resp.json()
    if not data.get("ok"):
        return False, f"Slack API error: {data.get('error', resp.status_code)}"
    return True, f"Alert posted to Slack channel {channel}."
