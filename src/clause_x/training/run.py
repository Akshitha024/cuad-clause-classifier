"""Training loop using the HF Trainer.

Intentionally pinned to CPU-friendly defaults (small batch, gradient
accumulation, no fp16, no DDP). On a MacBook the small Legal-BERT trains
3 epochs on a few thousand snippets in roughly 10-20 minutes; bigger
machines should bump the batch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from ..model.lora import LoRAConfig, build_model


@dataclass
class TrainConfig:
    data_dir: Path
    out_dir: Path
    base_model: str = "nlpaueb/legal-bert-small-uncased"
    epochs: int = 3
    batch_size: int = 8
    grad_accum: int = 2
    learning_rate: float = 2e-4
    max_length: int = 256
    eval_strategy: str = "epoch"
    seed: int = 7


def train(cfg: TrainConfig) -> Path:
    from datasets import Dataset
    from transformers import (
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    train_rows = _read_jsonl(cfg.data_dir / "train.jsonl")
    val_rows = _read_jsonl(cfg.data_dir / "val.jsonl")
    if not train_rows or not val_rows:
        raise RuntimeError(f"empty train/val under {cfg.data_dir}; run `clause-x data prepare`")

    logger.info("train rows: {}, val rows: {}", len(train_rows), len(val_rows))

    tokenizer, model = build_model(LoRAConfig(base_model=cfg.base_model, num_labels=2))

    def encode(row: dict[str, Any]) -> dict[str, Any]:
        # the clause is prepended so the model is conditional on which clause it's testing
        text = f"[{row['clause']}] {row['text']}"
        out: dict[str, Any] = tokenizer(text, truncation=True, max_length=cfg.max_length)
        out["labels"] = int(row["label"])
        return out

    train_ds = Dataset.from_list(train_rows).map(encode, remove_columns=list(train_rows[0].keys()))
    val_ds = Dataset.from_list(val_rows).map(encode, remove_columns=list(val_rows[0].keys()))

    args = TrainingArguments(
        output_dir=str(cfg.out_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        eval_strategy=cfg.eval_strategy,
        save_strategy="no",
        logging_steps=20,
        seed=cfg.seed,
        report_to=[],  # no wandb/tensorboard
        fp16=False,
        bf16=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    final = trainer.evaluate()
    metrics_path = cfg.out_dir / "final_metrics.json"
    metrics_path.write_text(json.dumps(final, indent=2))
    logger.info("final metrics written to {}", metrics_path)
    # save LoRA adapter
    model.save_pretrained(cfg.out_dir / "lora-adapter")
    tokenizer.save_pretrained(cfg.out_dir / "lora-adapter")
    return cfg.out_dir


def _read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _compute_metrics(eval_pred: tuple[Any, Any]) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_binary": float(f1_score(labels, preds, average="binary", zero_division=0)),
    }
