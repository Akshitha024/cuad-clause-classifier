from __future__ import annotations

from clause_x.model.lora import LoRAConfig


def test_default_targets() -> None:
    cfg = LoRAConfig()
    assert "query" in cfg.target_modules
    assert "value" in cfg.target_modules


def test_rank_alpha_sane() -> None:
    cfg = LoRAConfig()
    assert cfg.rank > 0
    assert cfg.alpha >= cfg.rank


def test_overrides() -> None:
    cfg = LoRAConfig(base_model="bert-base-uncased", rank=4, alpha=8)
    assert cfg.base_model == "bert-base-uncased"
    assert cfg.rank == 4
    assert cfg.alpha == 8
