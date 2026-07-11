from pathlib import Path
import numpy as np
from rugby_value.epa import observed_epa
from rugby_value.model import MarkovEPV
from rugby_value.preprocess import load_phase_data, prepare_trajectories
from rugby_value.schema import OUTCOME_REWARD

FIXTURE = Path(__file__).parent / "fixtures" / "phases.csv"

def test_epa_telescopes_to_terminal_reward_minus_start_value() -> None:
    trajectories = prepare_trajectories(load_phase_data(FIXTURE))
    model = MarkovEPV().fit(trajectories)
    epa = observed_epa(model, trajectories)
    for trajectory in trajectories:
        group = epa[epa.possession_id == trajectory.possession_id]
        expected = OUTCOME_REWARD[trajectory.outcome] - model.value(trajectory.states[0])
        assert np.isclose(group.epa.sum(), expected)
