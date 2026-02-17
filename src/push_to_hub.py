"""
Push trained model and tokenizer to HuggingFace Hub.
"""

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from huggingface_hub import HfApi, login

from .utils import MODEL_NAME, CACHED_MODEL_DIR


def push_model_to_hub(model_dir=CACHED_MODEL_DIR,
                      repo_name="distilbert-goodreads-genre-classifier",
                      hf_username="kingkenche",
                      hf_token=None):
    """
    Push the fine-tuned model, tokenizer, and config to HuggingFace Hub.

    Args:
        model_dir: Local directory containing the saved model.
        repo_name: Name of the HF repo to create/update.
        hf_username: HuggingFace username.
        hf_token: HuggingFace write access token.
    """
    if hf_token:
        login(token=hf_token)

    repo_id = f"{hf_username}/{repo_name}"
    print(f"Pushing model to HuggingFace Hub: {repo_id}")

    # Load saved model and tokenizer
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    # Push to hub
    model.push_to_hub(repo_id, token=hf_token)
    tokenizer.push_to_hub(repo_id, token=hf_token)

    print(f"Model pushed successfully → https://huggingface.co/{repo_id}")
    return repo_id


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Push model to HuggingFace Hub")
    parser.add_argument("--model-dir", default=CACHED_MODEL_DIR)
    parser.add_argument("--repo-name", default="distilbert-goodreads-genre-classifier")
    parser.add_argument("--hf-username", default="kingkenche")
    parser.add_argument("--hf-token", required=True, help="HuggingFace write token")
    args = parser.parse_args()

    push_model_to_hub(
        model_dir=args.model_dir,
        repo_name=args.repo_name,
        hf_username=args.hf_username,
        hf_token=args.hf_token,
    )
