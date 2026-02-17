#!/usr/bin/env python3
"""
eval_only.py — Production entry point for the eval-only Docker container (Task 9).

This script:
  1. Pulls the fine-tuned model from HuggingFace Hub
  2. Downloads and prepares Goodreads test data
  3. Runs evaluation and prints results
"""

import os
import random
import json

random.seed(42)

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments
from src.data import load_all_genres, split_data
from src.utils import encode_labels, ReviewDataset, compute_metrics, DEVICE
from src.eval import evaluate_model, save_results


def main():
    repo_id = os.environ.get("HF_REPO_ID", "kingkenche/distilbert-goodreads-genre-classifier")

    print("=" * 60)
    print("  EVALUATION-ONLY CONTAINER")
    print(f"  Model: {repo_id}")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading Goodreads reviews...")
    genre_reviews = load_all_genres()
    train_texts, train_labels, test_texts, test_labels = split_data(genre_reviews)
    label2id, id2label = encode_labels(train_labels)

    # ── Load model from HuggingFace ────────────────────────────────────────
    print(f"\n[2/4] Loading model from HuggingFace: {repo_id}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(repo_id)
    model = DistilBertForSequenceClassification.from_pretrained(repo_id).to(DEVICE)

    # ── Tokenize test data ─────────────────────────────────────────────────
    print("\n[3/4] Tokenizing test data...")
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)
    test_labels_enc = [label2id[y] for y in test_labels]
    test_dataset = ReviewDataset(test_encodings, test_labels_enc)

    # ── Evaluate ───────────────────────────────────────────────────────────
    print("\n[4/4] Running evaluation...")
    eval_args = TrainingArguments(
        output_dir="./results_eval",
        per_device_eval_batch_size=16,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        compute_metrics=compute_metrics,
        eval_dataset=test_dataset,
    )

    results, report, predicted_labels = evaluate_model(trainer, test_dataset, id2label, test_labels)
    save_results(results, "results/eval_results_docker.json")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1 Score:  {results['f1']:.4f}")
    print(f"  Loss:      {results['eval_loss']:.4f}")
    print("=" * 60)
    print("  Evaluation complete ✓")


if __name__ == "__main__":
    main()
