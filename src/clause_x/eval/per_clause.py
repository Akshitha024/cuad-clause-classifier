"""Per-clause evaluation: run the trained model on the val set, compute F1
per clause, write a sorted summary.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from loguru import logger
from sklearn.metrics import f1_score, precision_score, recall_score


def score(adapter_dir: Path, data_dir: Path, out_dir: Path) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)  # type: ignore[no-untyped-call]
    cfg_path = adapter_dir / "adapter_config.json"
    base = json.loads(cfg_path.read_text())["base_model_name_or_path"]
    base_model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    val_rows = [
        json.loads(line) for line in (data_dir / "val.jsonl").read_text().splitlines() if line
    ]
    by_clause_preds: dict[str, list[int]] = defaultdict(list)
    by_clause_gold: dict[str, list[int]] = defaultdict(list)

    with torch.no_grad():
        for row in val_rows:
            text = f"[{row['clause']}] {row['text']}"
            enc = tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
            logits = model(**enc).logits
            pred = int(torch.argmax(logits, dim=-1).item())
            by_clause_preds[row["clause"]].append(pred)
            by_clause_gold[row["clause"]].append(int(row["label"]))

    rows = []
    for clause in sorted(by_clause_preds.keys()):
        preds = by_clause_preds[clause]
        gold = by_clause_gold[clause]
        rows.append(
            {
                "clause": clause,
                "n": len(gold),
                "precision": float(precision_score(gold, preds, zero_division=0)),
                "recall": float(recall_score(gold, preds, zero_division=0)),
                "f1": float(f1_score(gold, preds, zero_division=0)),
                "support_positive": int(sum(gold)),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "per_clause.json"
    summary.write_text(json.dumps(rows, indent=2))
    logger.info("wrote {}", summary)
    return summary
