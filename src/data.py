"""
Data loading and preprocessing for Goodreads book reviews.
"""

import gzip
import json
import random
import requests
import pickle
import os

from .utils import GENRE_URL_DICT


def load_reviews(url, head=10000, sample_size=2000):
    """
    Stream reviews from a gzipped JSON URL and return a random sample.

    Args:
        url: URL to a .json.gz file of Goodreads reviews.
        head: Maximum number of reviews to read from the stream.
        sample_size: Number of reviews to randomly sample from the loaded set.

    Returns:
        List of review text strings.
    """
    reviews = []
    count = 0

    response = requests.get(url, stream=True)
    response.raise_for_status()
    print(f"  HTTP {response.status_code} — streaming reviews...")

    with gzip.open(response.raw, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            reviews.append(d["review_text"])
            count += 1
            if head is not None and count >= head:
                break

    return random.sample(reviews, min(sample_size, len(reviews)))


def load_all_genres(genre_url_dict=None, head=10000, sample_size=2000,
                    cache_path="genre_reviews_dict.pickle"):
    """
    Load reviews for every genre. Uses a pickle cache if available.

    Returns:
        dict[str, list[str]]: genre → list of review texts.
    """
    if genre_url_dict is None:
        genre_url_dict = GENRE_URL_DICT

    # Try loading from cache first
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached reviews from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    genre_reviews = {}
    for genre, url in genre_url_dict.items():
        print(f"Loading reviews for genre: {genre}")
        genre_reviews[genre] = load_reviews(url, head=head, sample_size=sample_size)

    # Cache for future runs
    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump(genre_reviews, f)
        print(f"Cached reviews to {cache_path}")

    return genre_reviews


def split_data(genre_reviews_dict, sample_per_genre=1000, train_size=800):
    """
    Split genre_reviews_dict into train and test sets.

    Args:
        genre_reviews_dict: dict[str, list[str]]
        sample_per_genre: Total reviews to use per genre.
        train_size: Number of training reviews per genre (rest → test).

    Returns:
        (train_texts, train_labels, test_texts, test_labels)
    """
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for genre, reviews in genre_reviews_dict.items():
        sampled = random.sample(reviews, min(sample_per_genre, len(reviews)))
        for review in sampled[:train_size]:
            train_texts.append(review)
            train_labels.append(genre)
        for review in sampled[train_size:]:
            test_texts.append(review)
            test_labels.append(genre)

    print(f"Train: {len(train_texts)} samples | Test: {len(test_texts)} samples")
    return train_texts, train_labels, test_texts, test_labels
