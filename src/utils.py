"""
Utility functions: constants, dataset class, label encoding, metrics.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-cased"
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHED_MODEL_DIR = "distilbert-reviews-genres"

GENRE_URL_DICT = {
    "poetry":                 "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
    "children":               "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz",
    "comics_graphic":         "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
    "fantasy_paranormal":     "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",
    "history_biography":      "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",
    "mystery_thriller_crime": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",
    "romance":                "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",
    "young_adult":            "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
}


# ── Custom PyTorch Dataset ─────────────────────────────────────────────────────
class ReviewDataset(torch.utils.data.Dataset):
    """Wraps tokenized encodings + integer labels into a PyTorch Dataset."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ── Label helpers ──────────────────────────────────────────────────────────────
def encode_labels(labels):
    """Return label2id and id2label dicts from a list of string labels."""
    unique = sorted(set(labels))
    label2id = {lbl: i for i, lbl in enumerate(unique)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return label2id, id2label


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(pred):
    """Compute accuracy, precision, recall, and F1 for the Trainer."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
