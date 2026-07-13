from pathlib import Path

import numpy as np

from rugby_value.epa import observed_epa
from rugby_value.model import CrossPossessionEPV
from rugby_value.preprocess import load_phase_data, prepare_steps

FIXTURE = Path(__file__).parent / "fixtures" / "phases.csv"


def test_epa_matches_signed_transition_definition() -> None:
    steps = prepare_steps(load_phase_data(FIXTURE))
    model = CrossPossessionEPV().fit(steps)
    frame = observed_epa(model, steps)
    assert len(frame) == len(steps)
    assert np.allclose(
        frame["epa"],
        frame["immediate_reward"]
        + frame["epv_after_from_source_perspective"]
        - frame["epv_before"],
    )
