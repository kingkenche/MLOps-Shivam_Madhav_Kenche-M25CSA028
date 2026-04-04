"""
upload_to_hf.py
===============
Upload trained model checkpoints to HuggingFace Hub.

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login          # paste your HF write token

Usage:
    python upload_to_hf.py --repo <your-hf-username>/<repo-name>

Example:
    python upload_to_hf.py --repo johndoe/assignment5-adversarial-art
"""

import argparse
import os
from huggingface_hub import HfApi, create_repo

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_FILES = {
    "resnet18_cifar10_best.pth": "checkpoints/resnet18_cifar10_best.pth",
    "detector_pgd.pth":          "checkpoints/detector_pgd.pth",
    "detector_bim.pth":          "checkpoints/detector_bim.pth",
}

RESULT_FILES = {
    "fgsm_comparison.png":    "results/fgsm_comparison.png",
    "fgsm_epsilon_curve.png": "results/fgsm_epsilon_curve.png",
    "evaluation_summary.png": "results/evaluation_summary.png",
}


def main():
    parser = argparse.ArgumentParser(description="Upload weights to HuggingFace Hub")
    parser.add_argument("--repo", required=True,
                        help="HuggingFace repo id, e.g. username/repo-name")
    parser.add_argument("--private", action="store_true",
                        help="Make the HF repo private (default: public)")
    args = parser.parse_args()

    api = HfApi()

    # Create repo if it doesn't exist
    print(f"Creating / connecting to HF repo: {args.repo}")
    create_repo(repo_id=args.repo, repo_type="model",
                private=args.private, exist_ok=True)

    # Upload checkpoints
    print("\nUploading model weights …")
    for hf_name, local_path in CHECKPOINT_FILES.items():
        if not os.path.exists(local_path):
            print(f"  SKIP  {local_path}  (not found – run training first)")
            continue
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"checkpoints/{hf_name}",
            repo_id=args.repo,
            repo_type="model",
        )
        print(f"  ✓  {local_path}  →  {args.repo}/checkpoints/{hf_name}")

    # Upload result images
    print("\nUploading result images …")
    for hf_name, local_path in RESULT_FILES.items():
        if not os.path.exists(local_path):
            print(f"  SKIP  {local_path}  (not found – run evaluation first)")
            continue
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"results/{hf_name}",
            repo_id=args.repo,
            repo_type="model",
        )
        print(f"  ✓  {local_path}  →  {args.repo}/results/{hf_name}")

    # Upload a minimal model card
    model_card = f"""---
license: mit
tags:
  - adversarial-robustness
  - cifar10
  - pytorch
  - ibm-art
---

# Assignment 5 – Adversarial Attacks & Detection (Q2)

## Overview
- **ResNet-18** trained from scratch on CIFAR-10 (≥ 72% clean accuracy).
- **FGSM** attack implemented from scratch and via IBM ART.
- **ResNet-34** binary detectors trained on PGD and BIM adversarial examples.

## Checkpoints
| File | Description |
|------|-------------|
| `checkpoints/resnet18_cifar10_best.pth` | Best ResNet-18 classifier |
| `checkpoints/detector_pgd.pth` | ResNet-34 PGD detector |
| `checkpoints/detector_bim.pth` | ResNet-34 BIM detector |

## Results
See `results/` folder and the WandB project linked in the GitHub README.
"""
    model_card_path = "/tmp/README_hf.md"
    with open(model_card_path, "w") as f:
        f.write(model_card)
    api.upload_file(
        path_or_fileobj=model_card_path,
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
    )
    print(f"\n✓  Model card uploaded.")
    print(f"\nHuggingFace repo URL: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
