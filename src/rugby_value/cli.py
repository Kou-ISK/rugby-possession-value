from __future__ import annotations

from pathlib import Path

import typer

from .datasets import fetch_phase_data
from .epa import observed_epa
from .model import CrossPossessionEPV, ModelConfig
from .outputs import write_tables
from .preprocess import load_phase_data, prepare_steps

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
) -> None:
    frame = load_phase_data(input_path)
    steps = prepare_steps(frame)
    model = CrossPossessionEPV(ModelConfig()).fit(steps)
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "model.json")
    write_tables(model, output)
    observed_epa(model, steps).to_csv(output / "observed_epa.csv", index=False)
    typer.echo(f"states={len(model.states)} steps={len(steps)}")


@app.command()
def table(
    model_path: Path,
    output: Path = typer.Option(Path("data/processed"), "--output", "-o"),
) -> None:
    model = CrossPossessionEPV.load(model_path)
    write_tables(model, output)
    typer.echo(output)


if __name__ == "__main__":
    app()
