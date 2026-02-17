"""
Evaluation: classification report, confusion matrix, result saving.
"""

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Docker / headless
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support


def evaluate_model(trainer, test_dataset, id2label, test_labels):
    """
    Run evaluation and return a results dict + classification report string.

    Returns:
        (results_dict, report_str, predicted_labels)
    """
    # Trainer built-in eval
    eval_result = trainer.evaluate()

    # Detailed predictions
    predictions = trainer.predict(test_dataset)
    pred_ids = predictions.predictions.argmax(-1).flatten().tolist()
    predicted_labels = [id2label[i] for i in pred_ids]

    # Classification report
    report_str = classification_report(test_labels, predicted_labels)
    print("\n=== Classification Report ===")
    print(report_str)

    # Compute overall metrics
    acc = accuracy_score(test_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predicted_labels, average="weighted", zero_division=0
    )

    results_dict = {
        "eval_loss": eval_result.get("eval_loss"),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report_str,
    }

    return results_dict, report_str, predicted_labels


def save_results(results_dict, filepath="results/eval_results.json"):
    """Save evaluation results to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results saved to {filepath}")


def generate_confusion_matrix(test_labels, predicted_labels,
                               output_path="results/confusion_matrix.png"):
    """Generate and save a confusion-matrix heatmap."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    genre_counts = defaultdict(int)
    for true, pred in zip(test_labels, predicted_labels):
        genre_counts[(true, pred)] += 1

    rows = []
    for (true_g, pred_g), count in genre_counts.items():
        rows.append({
            "True Genre": true_g,
            "Predicted Genre": pred_g,
            "Count": count,
        })

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="True Genre", columns="Predicted Genre", values="Count")

    plt.figure(figsize=(9, 7))
    sns.set(style="ticks", font_scale=1.2)
    sns.heatmap(pivot, linewidths=1, cmap="Purples", annot=True, fmt=".0f")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def generate_misclassification_matrix(test_labels, predicted_labels,
                                       output_path="results/misclassification_matrix.png"):
    """Confusion matrix with diagonal removed to highlight misclassifications."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    genre_counts = defaultdict(int)
    for true, pred in zip(test_labels, predicted_labels):
        if true != pred:
            genre_counts[(true, pred)] += 1

    if not genre_counts:
        print("No misclassifications found — skipping misclassification matrix.")
        return

    rows = []
    for (true_g, pred_g), count in genre_counts.items():
        rows.append({
            "True Genre": true_g,
            "Predicted Genre": pred_g,
            "Count": count,
        })

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="True Genre", columns="Predicted Genre", values="Count")

    plt.figure(figsize=(9, 7))
    sns.set(style="ticks", font_scale=1.2)
    sns.heatmap(pivot, linewidths=1, cmap="Purples", annot=True, fmt=".0f")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Misclassification matrix saved to {output_path}")
