from pathlib import Path

import numpy as np
import pandas as pd

from rugby_value.model import CrossPossessionEPV
from rugby_value.preprocess import load_phase_data, prepare_steps

FIXTURE = Path(__file__).parent / "fixtures" / "phases.csv"


def test_repeated_match_data_produces_valid_probabilities() -> None:
    base = load_phase_data(FIXTURE)
    frames = []
    for index in range(6):
        copy = base.copy()
        copy["Round"] = copy["Round"] + index * 10
        copy["ID"] = copy["ID"] + index * 100
        frames.append(copy)
    steps = prepare_steps(pd.concat(frames, ignore_index=True))
    model = CrossPossessionEPV().fit(steps)
    assert model.absorption is not None
    assert np.allclose(model.absorption.sum(axis=1), 1.0)
    assert np.isfinite(model.values).all()
