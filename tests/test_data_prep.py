from __future__ import annotations

import random

from clause_x.data.cuad import _extract_clause, _snippets_for


def test_extract_clause_quoted() -> None:
    q = 'Highlight the parts of this contract related to "Document Name" that should be reviewed.'
    assert _extract_clause(q) == "Document Name"


def test_extract_clause_no_quotes_returns_empty() -> None:
    assert _extract_clause("highlight the relevant section") == ""


def test_snippets_for_absent_clause_returns_random_windows() -> None:
    body = "x" * 10_000
    snips = _snippets_for(rows=[], body=body, present=False, rng=random.Random(0))
    assert len(snips) == 2
    assert all(0 < len(s) <= 1200 for s in snips)


def test_snippets_for_present_uses_answer_spans() -> None:
    body = "alpha " * 500 + "DAMAGES CAP: $100,000. " + "omega " * 500
    rows = [{"answers": {"text": ["DAMAGES CAP: $100,000."], "answer_start": [3000]}}]
    snips = _snippets_for(rows=rows, body=body, present=True, rng=random.Random(0))
    assert len(snips) >= 1
    # at least one snippet should contain the answer text
    assert any("DAMAGES CAP" in s for s in snips)


def test_snippets_for_short_body_just_returns_body() -> None:
    body = "tiny body"
    snips = _snippets_for(rows=[], body=body, present=False, rng=random.Random(0))
    assert "tiny body" in snips
