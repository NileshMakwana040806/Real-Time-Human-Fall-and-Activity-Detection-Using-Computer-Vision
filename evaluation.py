import csv
from collections import defaultdict


LABELS = ["STANDING", "WALKING", "SITTING", "FALL", "UNKNOWN"]


def compute_accuracy(y_true, y_pred):
    if not y_true:
        return 0.0
    correct = sum(1 for expected, predicted in zip(y_true, y_pred) if expected == predicted)
    return correct / len(y_true)


def confusion_matrix(y_true, y_pred, labels=None):
    labels = labels or LABELS
    matrix = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for actual, predicted in zip(y_true, y_pred):
        if actual not in matrix:
            matrix[actual] = defaultdict(int)
        matrix[actual][predicted] += 1
    return matrix


def evaluate_csv(csv_path, labels=None):
    y_true = []
    y_pred = []

    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            y_true.append(row["actual"].strip().upper())
            y_pred.append(row["predicted"].strip().upper())

    labels = labels or sorted(set(LABELS + y_true + y_pred))
    return {
        "accuracy": compute_accuracy(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels),
        "samples": len(y_true),
        "labels": labels,
    }
