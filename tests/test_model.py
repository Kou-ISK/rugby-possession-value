from pathlib import Path
import numpy as np
from rugby_value.model import MarkovEPV
from rugby_value.preprocess import load_phase_data, prepare_trajectories

FIXTURE = Path(__file__).parent / "fixtures" / "phases.csv"

def fitted() -> MarkovEPV:
    return MarkovEPV().fit(prepare_trajectories(load_phase_data(FIXTURE)))

def test_transition_rows_sum_to_one() -> None:
    model = fitted()
    assert model.q is not None and model.b is not None
    assert np.allclose(model.q.sum(axis=1) + model.b.sum(axis=1), 1.0)

def test_absorption_probabilities_sum_to_one() -> None:
    model = fitted()
    assert model.absorption is not None
    assert np.allclose(model.absorption.sum(axis=1), 1.0)

def test_model_round_trip(tmp_path: Path) -> None:
    model = fitted()
    path = tmp_path / "model.json"
    model.save(path)
    loaded = MarkovEPV.load(path)
    assert np.allclose(model.values, loaded.values)
