"""Dataset-free smoke checks for the CrossSense AI Streamlit application."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["Overview", "Industrial Fan", "Industrial Bearing", "CNC Milling Machine"]


def open_page(page):
    app = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=300)
    app.run()
    assert not app.exception, [error.value for error in app.exception]
    if page != "Overview":
        app.radio[0].set_value(page).run()
    return app


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    app = open_page(page)
    assert not app.exception, [error.value for error in app.exception]


def test_overview_lists_all_machines():
    app = open_page("Overview")
    headings = {heading.value for heading in app.subheader}
    assert {"Industrial Fan (MIMII DUE)", "Industrial Bearing (NASA IMS)",
            "CNC Milling (MSM Archive)"} <= headings
    labels = [metric.label for metric in app.metric]
    assert "Overall Plant Health" in labels and "Active Machine Alerts" in labels


def test_cnc_archive_scores_an_interval():
    app = open_page("CNC Milling Machine")
    assert [metric.label for metric in app.metric] == ["Anomaly score"]
    assert app.metric[0].value
    machines = app.selectbox[1].options
    assert {"imi_vm20i", "imi_vmx30ui", "tmf_vf10"} <= set(machines)


def test_cnc_switches_machine_configuration():
    app = open_page("CNC Milling Machine")
    app.selectbox[1].set_value("tmf_vf10").run()
    assert not app.exception, [error.value for error in app.exception]
    assert app.metric[0].value
