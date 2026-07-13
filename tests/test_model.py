from pathlib import Path

import numpy as np

from rugby_value.model import CrossPossessionEPV
from rugby_value.preprocess import load_phase_data, prepare_steps

FIXTURE = Path(__file__).parent / "fixtures" / "phases.csv"


def fitted() -> CrossPossessionEPV:
    return CrossPossessionEPV().fit(prepare_steps(load_phase_data(FIXTURE)))


def test_transition_rows_sum_to_one() -> None:
    model = fitted()
    assert model.p_same is not None
    assert model.p_opp is not None
    assert model.p_absorb is not None
    assert np.allclose(
        model.p_same.sum(axis=1)
        + model.p_opp.sum(axis=1)
        + model.p_absorb.sum(axis=1),
        1.0,
    )


def test_absorption_probabilities_sum_to_one() -> None:
    model = fitted()
    assert model.absorption is not None
    assert np.allclose(model.absorption.sum(axis=1), 1.0)


def test_model_round_trip(tmp_path: Path) -> None:
    model = fitted()
    path = tmp_path / "model.json"
    model.save(path)
    loaded = CrossPossessionEPV.load(path)
    assert np.allclose(model.values, loaded.values)
