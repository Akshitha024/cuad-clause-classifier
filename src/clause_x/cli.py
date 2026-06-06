from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .data.cuad import build as build_data
from .eval.per_clause import score as per_clause_score
from .eval.plots import plot_per_clause
from .training.run import TrainConfig, train

app = typer.Typer(add_completion=False, help="clause-x: LoRA clause classifier")
data_grp = typer.Typer(help="dataset prep")
train_grp = typer.Typer(help="training")
score_grp = typer.Typer(help="evaluation")
app.add_typer(data_grp, name="data")
app.add_typer(train_grp, name="train")
app.add_typer(score_grp, name="eval")


@data_grp.command("prepare")
def cmd_data(
    out: Annotated[Path, typer.Option(help="dataset dir")] = Path("data/processed"),
    limit: Annotated[int | None, typer.Option(help="cap examples; smoke runs")] = None,
) -> None:
    build_data(out, limit=limit)


@train_grp.command("run")
def cmd_train(
    data_dir: Annotated[Path, typer.Option(help="dataset dir")] = Path("data/processed"),
    out_dir: Annotated[Path, typer.Option(help="checkpoint dir")] = Path("checkpoints"),
    base_model: Annotated[str, typer.Option(help="encoder")] = "nlpaueb/legal-bert-small-uncased",
    epochs: Annotated[int, typer.Option(help="epochs")] = 3,
    batch_size: Annotated[int, typer.Option(help="per-device train batch")] = 8,
) -> None:
    cfg = TrainConfig(
        data_dir=data_dir,
        out_dir=out_dir,
        base_model=base_model,
        epochs=epochs,
        batch_size=batch_size,
    )
    train(cfg)


@score_grp.command("run")
def cmd_score(
    adapter_dir: Annotated[Path, typer.Option(help="lora adapter dir")] = Path(
        "checkpoints/lora-adapter"
    ),
    data_dir: Annotated[Path, typer.Option(help="dataset dir")] = Path("data/processed"),
    out_dir: Annotated[Path, typer.Option(help="results dir")] = Path("results"),
) -> None:
    summary = per_clause_score(adapter_dir, data_dir, out_dir)
    typer.echo(f"wrote {summary}")


@app.command("plots")
def cmd_plots(
    summary: Annotated[Path, typer.Option(help="per_clause.json")] = Path(
        "results/per_clause.json"
    ),
    out: Annotated[Path, typer.Option(help="figure path")] = Path(
        "results/figures/per_clause_f1.png"
    ),
) -> None:
    plot_per_clause(summary, out)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
