"""LoRA-wrapped sequence classifier on top of a Legal-BERT-family encoder.

Default base is ``nlpaueb/legal-bert-small-uncased`` (35M params, small enough
to fine-tune on CPU within an hour). LoRA targets the attention Q/V matrices
(the original LoRA paper's recommended starting set) with rank 8.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class LoRAConfig:
    base_model: str = "nlpaueb/legal-bert-small-uncased"
    num_labels: int = 2
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("query", "value")


def build_model(cfg: LoRAConfig) -> tuple[Any, Any]:
    """Return (tokenizer, lora_model). Caller decides what device to move to."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)  # type: ignore[no-untyped-call]
    base = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model, num_labels=cfg.num_labels
    )
    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
    )
    model = get_peft_model(base, peft_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA model ready: {:,} trainable / {:,} total ({:.2%})",
        trainable,
        total,
        trainable / total,
    )
    return tokenizer, model
