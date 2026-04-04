"""
model.py - ViT-S model definitions for baseline (no LoRA) and LoRA fine-tuning
"""

import timm
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


NUM_CLASSES = 100


# ---------------------------------------------------------------------------
# Helper: count trainable parameters
# ---------------------------------------------------------------------------

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# 1.  Baseline – only the classification head is fine-tuned
# ---------------------------------------------------------------------------

def get_baseline_vit(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """
    Load ViT-S/16 pre-trained on ImageNet-21k (via timm).
    Freeze all parameters, then replace + unfreeze the head.
    """
    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
    )

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the classification head
    for param in model.head.parameters():
        param.requires_grad = True

    trainable = count_trainable_params(model)
    total     = count_all_params(model)
    print(f"[Baseline] Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")
    return model


# ---------------------------------------------------------------------------
# 2.  LoRA fine-tuning via PEFT
# ---------------------------------------------------------------------------

def get_lora_vit(
    rank: int       = 4,
    alpha: float    = 4,
    dropout: float  = 0.1,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    target_modules: list = None,
):
    """
    Load ViT-S/16, apply LoRA to Q, K, V projections, and keep the
    classification head trainable.

    Args:
        rank:           LoRA rank r
        alpha:          LoRA scaling alpha
        dropout:        LoRA dropout
        num_classes:    output classes
        pretrained:     load ImageNet weights
        target_modules: list of module name patterns to inject LoRA into.
                        Defaults to query, key, value projections.
    """
    # timm ViT-S uses a single fused nn.Linear named "qkv" inside each
    # attention block (blocks.N.attn.qkv).  PEFT matches by substring, so
    # targeting "qkv" covers all 12 attention blocks (Q, K, V fused).
    if target_modules is None:
        target_modules = ["qkv"]   # fused Q+K+V linear in every attention block

    base_model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
    )

    lora_config = LoraConfig(
        r              = rank,
        lora_alpha     = alpha,
        lora_dropout   = dropout,
        target_modules = target_modules,
        bias           = "none",
        # task_type is not set (None) for ViT/custom models; PEFT will
        # wrap the targeted linear layers regardless.
    )

    model = get_peft_model(base_model, lora_config)

    # Make sure the classification head is also trainable
    for name, param in model.named_parameters():
        if "head" in name:
            param.requires_grad = True

    trainable = count_trainable_params(model)
    total     = count_all_params(model)
    print(f"[LoRA r={rank}, α={alpha}, drop={dropout}] "
          f"Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# 3.  Partially frozen LoRA (Optional Q3)
# ---------------------------------------------------------------------------

def get_partial_lora_vit(
    rank: int        = 4,
    alpha: float     = 4,
    dropout: float   = 0.1,
    num_classes: int = NUM_CLASSES,
    freeze_first_n_blocks: int = 6,
    pretrained: bool = True,
):
    """
    Freeze the first `freeze_first_n_blocks` transformer blocks completely,
    keep the remaining blocks fully trainable, and apply LoRA only to the
    frozen portion.  The classification head is always trainable.
    """
    base_model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
    )

    # Determine which attention modules belong to the frozen blocks
    target_modules = [
        f"blocks.{i}.attn.qkv" for i in range(freeze_first_n_blocks)
    ]

    # Freeze first N blocks
    for i in range(freeze_first_n_blocks):
        for param in base_model.blocks[i].parameters():
            param.requires_grad = False

    lora_config = LoraConfig(
        r              = rank,
        lora_alpha     = alpha,
        lora_dropout   = dropout,
        target_modules = target_modules,
        bias           = "none",
    )

    model = get_peft_model(base_model, lora_config)

    # Head always trainable
    for name, param in model.named_parameters():
        if "head" in name:
            param.requires_grad = True

    trainable = count_trainable_params(model)
    total     = count_all_params(model)
    print(f"[Partial LoRA r={rank}, α={alpha}, frozen_blocks={freeze_first_n_blocks}] "
          f"Trainable: {trainable:,} / {total:,}")
    return model
