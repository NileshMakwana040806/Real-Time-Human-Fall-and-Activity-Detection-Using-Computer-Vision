"""
evaluation.py — Metrics for activity detection: accuracy, confusion matrix,
                precision, recall, F1-score per class.
"""

import csv
from collections import defaultdict

LABELS = ["STANDING", "WALKING", "SITTING", "FALL", "UNKNOWN"]


# ─── Core metrics ─────────────────────────────────────────────────────────────

def compute_accuracy(y_true, y_pred):
    if not y_true:
        return 0.0
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def confusion_matrix(y_true, y_pred, labels=None):
    labels = labels or LABELS
    matrix = {a: {p: 0 for p in labels} for a in labels}
    for actual, predicted in zip(y_true, y_pred):
        if actual not in matrix:
            matrix[actual] = defaultdict(int)
        matrix[actual][predicted] = matrix[actual].get(predicted, 0) + 1
    return matrix


def compute_per_class_metrics(y_true, y_pred, labels=None):
    """Return precision, recall, F1 per class."""
    labels = labels or sorted(set(y_true + y_pred))
    results = {}
    for label in labels:
        tp = sum(1 for a, p in zip(y_true, y_pred) if a == label and p == label)
        fp = sum(1 for a, p in zip(y_true, y_pred) if a != label and p == label)
        fn = sum(1 for a, p in zip(y_true, y_pred) if a == label and p != label)

        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-9)
        results[label] = {"precision": round(precision, 4),
                          "recall":    round(recall,    4),
                          "f1":        round(f1,        4),
                          "support":   tp + fn}
    return results


# ─── CSV evaluation ───────────────────────────────────────────────────────────

def evaluate_csv(csv_path, labels=None):
    """
    Evaluate from a CSV with columns: actual, predicted
    Returns dict with accuracy, confusion_matrix, per_class_metrics, samples.
    """
    y_true, y_pred = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true.append(row["actual"].strip().upper())
            y_pred.append(row["predicted"].strip().upper())

    labels = labels or sorted(set(LABELS + y_true + y_pred))
    return {
        "accuracy":           compute_accuracy(y_true, y_pred),
        "confusion_matrix":   confusion_matrix(y_true, y_pred, labels),
        "per_class_metrics":  compute_per_class_metrics(y_true, y_pred, labels),
        "samples":            len(y_true),
        "labels":             labels,
    }


# ─── Pretty-print helper ──────────────────────────────────────────────────────

def print_report(results):
    print(f"\nAccuracy: {results['accuracy']:.4f}  ({results['samples']} samples)\n")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 56)
    for label, m in results["per_class_metrics"].items():
        print(f"{label:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1']:>10.4f} {m['support']:>10}")
    print("\nConfusion Matrix:")
    labels = results["labels"]
    header = f"{'':>12}" + "".join(f"{l:>12}" for l in labels)
    print(header)
    for actual in labels:
        row = f"{actual:>12}" + "".join(
            f"{results['confusion_matrix'].get(actual, {}).get(pred, 0):>12}"
            for pred in labels
        )
        print(row)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python evaluation.py results.csv")
    else:
        r = evaluate_csv(sys.argv[1])
        print_report(r)
