from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from .bootstrap import BootstrapConfig, cluster_bootstrap
from .datasets import fetch_phase_data
from .epa import observed_epa
from .model import MarkovEPV, ModelConfig
from .outputs import write_tables
from .preprocess import load_phase_data, prepare_trajectories
from .validation import CrossValidationConfig, match_group_cross_validate

app = typer.Typer(no_args_is_help=True, help="Rugby Possession Value CLI")


@app.command("fetch-data")
def fetch_data(
    output: Path = typer.Option(Path("data/raw/phase_2018-19.csv"), "--output", "-o"),
    overwrite: bool = False,
) -> None:
    path = fetch_phase_data(output, overwrite=overwrite)
    typer.echo(path)


@app.command()
def fit(
    input_path: Path,
    output: Path = typer.Option(Path("models/premiership-2018-19"), "--output", "-o"),
    bootstrap: int = typer.Option(0, min=0),
    seed: int = 20260711,
) -> None:
    frame = load_phase_data(input_path)
    trajectories = prepare_trajectories(frame)
    config = ModelConfig()
    model = MarkovEPV(config).fit(trajectories)
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "model.json")
    intervals: pd.DataFrame | None = None
    if bootstrap:
        intervals = cluster_bootstrap(
            trajectories, config, BootstrapConfig(iterations=bootstrap, seed=seed)
        )
        intervals.to_csv(output / "bootstrap_intervals.csv", index=False)
    write_tables(model, output, intervals)
    observed_epa(model, trajectories).to_csv(output / "observed_epa.csv", index=False)
    typer.echo(f"states={len(model.states)} possessions={len(trajectories)}")


@app.command()
def table(
    model_path: Path,
    output: Path = typer.Option(Path("data/processed"), "--output", "-o"),
) -> None:
    model = MarkovEPV.load(model_path)
    write_tables(model, output)
    typer.echo(output)


@app.command()
def validate(
    input_path: Path,
    output: Path = typer.Option(Path("validation"), "--output", "-o"),
    folds: int = typer.Option(5, min=2),
    seed: int = 20260711,
) -> None:
    trajectories = prepare_trajectories(load_phase_data(input_path))
    summary, predictions = match_group_cross_validate(
        trajectories, ModelConfig(), CrossValidationConfig(folds=folds, seed=seed)
    )
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "summary.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    typer.echo(summary.to_string(index=False))


if __name__ == "__main__":
    app()
