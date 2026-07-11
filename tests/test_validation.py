from pathlib import Path

import pandas as pd

from rugby_value.model import ModelConfig
from rugby_value.preprocess import load_phase_data, prepare_trajectories
from rugby_value.validation import CrossValidationConfig, match_group_cross_validate

FIXTURE = Path(__file__).parent / "fixtures" / "phases.csv"


def test_match_group_cross_validation() -> None:
    base = load_phase_data(FIXTURE)
    frames = []
    for i in range(6):
        copy = base.copy()
        copy["Round"] = copy["Round"] + i * 10
        copy["ID"] = copy["ID"] + i * 100
        frames.append(copy)
    trajectories = prepare_trajectories(pd.concat(frames, ignore_index=True))
    summary, predictions = match_group_cross_validate(
        trajectories, ModelConfig(), CrossValidationConfig(folds=3, seed=1)
    )
    assert len(summary) == 1
    assert len(predictions) > 0
    assert summary.loc[0, "coverage"] > 0
