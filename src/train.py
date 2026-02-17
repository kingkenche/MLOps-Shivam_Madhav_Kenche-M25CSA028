"""
Model loading, training configuration, and training loop.
"""

import os
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from .utils import MODEL_NAME, MAX_LENGTH, DEVICE, CACHED_MODEL_DIR, ReviewDataset, compute_metrics


def get_tokenizer(model_name=MODEL_NAME):
    """Load the DistilBERT tokenizer."""
    print(f"Loading tokenizer: {model_name}")
    return DistilBertTokenizerFast.from_pretrained(model_name)


def get_model(model_name=MODEL_NAME, num_labels=8, id2label=None, label2id=None):
    """Load a pre-trained DistilBERT model for sequence classification."""
    print(f"Loading model: {model_name} (num_labels={num_labels}) → device={DEVICE}")
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    return model.to(DEVICE)


def tokenize_data(tokenizer, texts, max_length=MAX_LENGTH):
    """Tokenize a list of texts using the DistilBERT tokenizer."""
    return tokenizer(texts, truncation=True, padding=True, max_length=max_length)


def get_training_args(output_dir="./results", logging_dir="./logs"):
    """Return a TrainingArguments object with the notebook's configuration."""

    # Disable wandb logging
    os.environ["WANDB_DISABLED"] = "true"

    return TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=16,
        learning_rate=5e-5,
        warmup_steps=100,
        weight_decay=0.01,
        output_dir=output_dir,
        logging_dir=logging_dir,
        logging_steps=100,
        eval_strategy="steps",
        save_strategy="epoch",
        load_best_model_at_end=False,
        report_to=[],
    )


def create_trainer(model, training_args, train_dataset, eval_dataset):
    """Create the HuggingFace Trainer."""
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )


def train_and_save(trainer, save_dir=CACHED_MODEL_DIR):
    """Run training and save the fine-tuned model."""
    print("Starting training...")
    trainer.train()
    print(f"Saving model to {save_dir}")
    trainer.save_model(save_dir)
    print("Training complete ✓")
