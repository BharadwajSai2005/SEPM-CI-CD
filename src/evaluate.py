import json, sys

# Medical-grade thresholds: recall is non-negotiable
THRESHOLDS = {
    "auc_roc":  0.85,
    "recall":   0.80,   # must catch 80%+ of diseased patients
    "f1":       0.78,
    "accuracy": 0.78,
}

def validate_model(path="reports/metrics.json"):
    with open(path) as f:
        m = json.load(f)

    print(f"\n[GATE] Model: {m['model']}")
    print(f"{'Metric':<12} {'Value':>8}   {'Threshold':>10}   Status")
    print("-" * 50)

    failures = []
    for metric, threshold in THRESHOLDS.items():
        value  = m[metric]
        status = "PASS" if value >= threshold else "FAIL"
        print(f"{metric:<12} {value:>8.4f}   {threshold:>10.4f}   {status}")
        if value < threshold:
            failures.append(f"{metric} {value:.4f} < {threshold}")

    print("-" * 50)
    if failures:
        print(f"\n[BLOCKED] Pipeline halted — {len(failures)} gate(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print("\n[PASS] All quality gates met — proceeding to deployment.")

if __name__ == "__main__":
    validate_model()