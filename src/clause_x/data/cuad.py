"""Build a per-clause classification dataset from CUAD.

CUAD ships as SQuAD-style (theatticusproject/cuad-qa): one row per (contract,
clause-question) pair, with an extracted answer span when the clause is
present and an empty list when absent. For classification we collapse that
into:

  - For each unique clause-type (41 of them) and each contract, produce a
    snippet of the contract text and a binary label (1 if the clause is
    present in this contract, else 0).

Snippets are randomly sampled fixed-length windows from the contract body.
If the clause is present, half the snippets per (contract, clause) cover the
answer span (so the model has a chance to learn the clause's surface form),
the other half are random windows (negative-in-positive contracts).

The result is a balanced binary classifier *per clause*. Final task layout:
  X = (clause_label_id, snippet_text) -> y in {0, 1}
We fold the label id into the input as a prompt prefix so a single model
can multi-task across all 41 clauses without dragging in a label embedding.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger

_WINDOW = 1200  # chars per snippet; ~300 tokens for legal text


def build(out_dir: Path, limit: int | None = None, seed: int = 7) -> Path:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("install `datasets` (uv sync)") from e

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    logger.info("loading theatticusproject/cuad-qa")
    raw = load_dataset("theatticusproject/cuad-qa", split="train", trust_remote_code=True)

    # group rows by (title, clause)
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    contracts: dict[str, str] = {}
    for row in raw:
        title = str(row["title"]).strip()
        clause = _extract_clause(str(row["question"]))
        if not title or not clause:
            continue
        contracts.setdefault(title, str(row["context"]))
        by_pair[(title, clause)].append(row)

    pairs_written = 0
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    labels_path = out_dir / "labels.json"

    titles = sorted(contracts.keys())
    rng.shuffle(titles)
    split_at = max(1, int(0.8 * len(titles)))
    train_titles = set(titles[:split_at])

    clause_set: set[str] = {c for (_, c) in by_pair}
    label2id = {c: i for i, c in enumerate(sorted(clause_set))}
    labels_path.write_text(json.dumps(label2id, indent=2, sort_keys=True))

    with train_path.open("w") as ftr, val_path.open("w") as fva:
        for (title, clause), rows in by_pair.items():
            label_id = label2id[clause]
            body = contracts[title]
            present = any(_has_answer(r.get("answers")) for r in rows)
            for snippet in _snippets_for(rows, body, present, rng):
                obj = {
                    "title": title,
                    "clause": clause,
                    "clause_id": label_id,
                    "text": snippet,
                    "label": 1 if present else 0,
                }
                line = json.dumps(obj) + "\n"
                if title in train_titles:
                    ftr.write(line)
                else:
                    fva.write(line)
                pairs_written += 1
                if limit is not None and pairs_written >= limit:
                    break
            if limit is not None and pairs_written >= limit:
                break

    logger.info(
        "wrote {} examples across {} contracts and {} clauses to {}",
        pairs_written,
        len(contracts),
        len(label2id),
        out_dir,
    )
    return out_dir


def _has_answer(answers: object) -> bool:
    if not isinstance(answers, dict):
        return False
    texts = answers.get("text") or []
    return any(t for t in texts if isinstance(t, str) and t.strip())


def _extract_clause(question: str) -> str:
    """CUAD questions are templated. Pull the clause name out of the quotes."""
    # e.g. 'Highlight the parts ... related to "Document Name" that should be reviewed ...'
    start = question.find('"')
    if start < 0:
        return ""
    end = question.find('"', start + 1)
    if end < 0:
        return ""
    return question[start + 1 : end].strip()


def _snippets_for(
    rows: list[dict[str, Any]],
    body: str,
    present: bool,
    rng: random.Random,
) -> list[str]:
    """Return up to 4 snippets per (contract, clause).

    Half are centered on an answer span (when present), half are random
    windows from elsewhere in the document. For absent clauses, all snippets
    are random.
    """
    out: list[str] = []
    if present:
        for row in rows:
            ans = row.get("answers") or {}
            starts = ans.get("answer_start") or []
            for start in starts[:2]:
                lo = max(0, int(start) - _WINDOW // 2)
                hi = min(len(body), lo + _WINDOW)
                out.append(body[lo:hi].strip())
    # random windows
    n_random = max(0, 2 - len(out)) if present else 2
    for _ in range(n_random):
        if len(body) <= _WINDOW:
            out.append(body)
        else:
            start = rng.randint(0, len(body) - _WINDOW)
            out.append(body[start : start + _WINDOW].strip())
    return [s for s in out if s]
