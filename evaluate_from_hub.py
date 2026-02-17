#!/usr/bin/env python3
"""
evaluate_from_hub.py — Task 8: Load fine-tuned model from HuggingFace repo,
run evaluation, and compare with local results.
"""

import os
import json
import random
import argparse

random.seed(42)

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments
from src.data import load_all_genres, split_data
from src.utils import encode_labels, ReviewDataset, compute_metrics, DEVICE
from src.eval import evaluate_model, save_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model from HuggingFace Hub")
    parser.add_argument("--repo-id", default="kingkenche/distilbert-goodreads-genre-classifier",
                        help="HuggingFace repo ID (e.g. username/model-name)")
    parser.add_argument("--local-results", default="results/eval_results.json",
                        help="Path to local evaluation results for comparison")
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading Goodreads reviews...")
    genre_reviews = load_all_genres()
    train_texts, train_labels, test_texts, test_labels = split_data(genre_reviews)

    label2id, id2label = encode_labels(train_labels)

    # ── Load model from HuggingFace ────────────────────────────────────────
    print(f"\nLoading model from HuggingFace: {args.repo_id}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.repo_id)
    model = DistilBertForSequenceClassification.from_pretrained(args.repo_id).to(DEVICE)

    # ── Tokenize test data ─────────────────────────────────────────────────
    print("Tokenizing test data...")
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)
    test_labels_enc = [label2id[y] for y in test_labels]
    test_dataset = ReviewDataset(test_encodings, test_labels_enc)

    # ── Create a Trainer for evaluation only ───────────────────────────────
    eval_args = TrainingArguments(
        output_dir="./results_hub",
        per_device_eval_batch_size=16,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        compute_metrics=compute_metrics,
        eval_dataset=test_dataset,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATING MODEL FROM HUGGINGFACE HUB")
    print("=" * 60)
    hub_results, report, predicted_labels = evaluate_model(trainer, test_dataset, id2label, test_labels)
    save_results(hub_results, "results/eval_results_hub.json")

    # ── Compare with local results ─────────────────────────────────────────
    if os.path.exists(args.local_results):
        print("\n" + "=" * 60)
        print("COMPARISON: Local vs HuggingFace Hub Model")
        print("=" * 60)
        with open(args.local_results) as f:
            local_results = json.load(f)

        print(f"{'Metric':<15} {'Local':>10} {'Hub':>10} {'Diff':>10}")
        print("-" * 50)
        for metric in ["accuracy", "precision", "recall", "f1"]:
            local_val = local_results.get(metric, 0)
            hub_val = hub_results.get(metric, 0)
            diff = hub_val - local_val
            print(f"{metric:<15} {local_val:>10.4f} {hub_val:>10.4f} {diff:>+10.4f}")
    else:
        print(f"\nNo local results found at {args.local_results} — skipping comparison.")

    print("\nDone ✓")


if __name__ == "__main__":
    main()
