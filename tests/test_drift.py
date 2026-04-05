import pytest, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from monitor import run_drift_report

def test_drift_report_runs():
    summary = run_drift_report()
    assert "dataset_drift"    in summary
    assert "drifted_features" in summary
    assert isinstance(summary["n_drifted"], int)

def test_drift_report_file_created():
    run_drift_report()
    assert os.path.exists("reports/drift_summary.json")
    assert os.path.exists("reports/drift_report.html")

def test_drift_summary_schema():
    with open("reports/drift_summary.json") as f:
        s = json.load(f)
    assert isinstance(s["dataset_drift"],    bool)
    assert isinstance(s["drifted_features"], list)
    assert s["n_drifted"] == len(s["drifted_features"])