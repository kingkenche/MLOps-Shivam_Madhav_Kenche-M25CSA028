"""
push_to_hub.py
==============
Push the best model checkpoint(s) to HuggingFace Hub.

Usage:
    python push_to_hub.py \
        --ckpt checkpoints/lora_r4_a4_d0.1_best.pth \
        --mode lora \
        --rank 4 --alpha 4 --dropout 0.1 \
        --hf_repo YOUR_HF_USERNAME/vit-cifar100-lora \
        --hf_token YOUR_HF_TOKEN
"""

import argparse
import os
import torch
from huggingface_hub import HfApi, create_repo

from model import get_lora_vit, get_baseline_vit


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       type=str, required=True)
    p.add_argument("--mode",       type=str, default="lora",
                   choices=["baseline", "lora"])
    p.add_argument("--rank",       type=int,   default=4)
    p.add_argument("--alpha",      type=float, default=4.0)
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--hf_repo",    type=str, required=True,
                   help="HuggingFace repo id, e.g. username/vit-cifar100-lora")
    p.add_argument("--hf_token",   type=str, default=None,
                   help="HuggingFace API token (or set HF_TOKEN env var)")
    return p.parse_args()


def main():
    args  = parse_args()
    token = args.hf_token or os.environ.get("HF_TOKEN")
    api   = HfApi(token=token)

    # Create repo if it doesn't exist
    create_repo(args.hf_repo, token=token, exist_ok=True)

    # Upload checkpoint directly
    print(f"Uploading {args.ckpt} → {args.hf_repo}")
    api.upload_file(
        path_or_fileobj = args.ckpt,
        path_in_repo    = os.path.basename(args.ckpt),
        repo_id         = args.hf_repo,
        token           = token,
    )

    # Also upload a model card
    readme = f"""# ViT-S CIFAR-100 LoRA Fine-tuning

## Model Details
- **Base model**: `timm/vit_small_patch16_224` (ImageNet pre-trained)
- **Dataset**: CIFAR-100
- **Fine-tuning**: {'LoRA (PEFT)' if args.mode == 'lora' else 'Classification head only'}

{'### LoRA Config' if args.mode == 'lora' else ''}
{'- Rank: ' + str(args.rank) if args.mode == 'lora' else ''}
{'- Alpha: ' + str(args.alpha) if args.mode == 'lora' else ''}
{'- Dropout: ' + str(args.dropout) if args.mode == 'lora' else ''}
{'- Target modules: attn.qkv (Q, K, V)' if args.mode == 'lora' else ''}

## Usage
```python
import torch, timm
from peft import PeftModel
# Load the model and weights from this repo
```

## Assignment
CS/DS – Assignment 5 | ViT CIFAR-100 LoRA
"""
    card_path = "/tmp/README.md"
    with open(card_path, "w") as f:
        f.write(readme)

    api.upload_file(
        path_or_fileobj = card_path,
        path_in_repo    = "README.md",
        repo_id         = args.hf_repo,
        token           = token,
    )

    print(f"✓ Model pushed to https://huggingface.co/{args.hf_repo}")


if __name__ == "__main__":
    main()
