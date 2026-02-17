#!/usr/bin/env python3
"""
main.py — Full pipeline: load data → tokenize → train → evaluate → save.

This script is designed to run on Google Colab (with GPU) or locally.
"""

import os
import random

# Reproducibility
random.seed(42)

from src.data import load_all_genres, split_data
from src.utils import encode_labels, ReviewDataset, CACHED_MODEL_DIR
from src.train import get_tokenizer, get_model, tokenize_data, get_training_args, create_trainer, train_and_save
from src.eval import evaluate_model, save_results, generate_confusion_matrix, generate_misclassification_matrix


def main():
    # ── Step 1: Load data ──────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading Goodreads reviews...")
    print("=" * 60)
    genre_reviews = load_all_genres()

    # ── Step 2: Split data ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Splitting into train/test sets...")
    print("=" * 60)
    train_texts, train_labels, test_texts, test_labels = split_data(genre_reviews)

    # ── Step 3: Encode labels ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Encoding labels...")
    print("=" * 60)
    label2id, id2label = encode_labels(train_labels)
    print(f"Labels: {label2id}")

    # ── Step 4: Tokenize ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Tokenizing texts...")
    print("=" * 60)
    tokenizer = get_tokenizer()
    train_encodings = tokenize_data(tokenizer, train_texts)
    test_encodings = tokenize_data(tokenizer, test_texts)

    train_labels_enc = [label2id[y] for y in train_labels]
    test_labels_enc = [label2id[y] for y in test_labels]

    train_dataset = ReviewDataset(train_encodings, train_labels_enc)
    test_dataset = ReviewDataset(test_encodings, test_labels_enc)
    print(f"Train dataset: {len(train_dataset)} | Test dataset: {len(test_dataset)}")

    # ── Step 5: Load model ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Loading pre-trained DistilBERT model...")
    print("=" * 60)
    model = get_model(num_labels=len(id2label), id2label=id2label, label2id=label2id)

    # ── Step 6: Train ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Training the model...")
    print("=" * 60)
    training_args = get_training_args()
    trainer = create_trainer(model, training_args, train_dataset, test_dataset)
    train_and_save(trainer, save_dir=CACHED_MODEL_DIR)

    # Also save tokenizer alongside model for easy loading
    tokenizer.save_pretrained(CACHED_MODEL_DIR)

    # ── Step 7: Evaluate ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: Evaluating the model...")
    print("=" * 60)
    results, report, predicted_labels = evaluate_model(trainer, test_dataset, id2label, test_labels)

    # Save results
    save_results(results, "results/eval_results.json")
    generate_confusion_matrix(test_labels, predicted_labels, "results/confusion_matrix.png")
    generate_misclassification_matrix(test_labels, predicted_labels, "results/misclassification_matrix.png")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE ✓")
    print(f"  Model saved to: {CACHED_MODEL_DIR}/")
    print(f"  Results saved to: results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
