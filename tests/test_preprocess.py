from pathlib import Path
from rugby_value.preprocess import load_phase_data, prepare_trajectories

FIXTURE = Path(__file__).parent / "fixtures" / "phases.csv"

def test_reconstructs_possessions() -> None:
    trajectories = prepare_trajectories(load_phase_data(FIXTURE))
    assert len(trajectories) == 4
    assert [len(t.states) for t in trajectories] == [3, 1, 2, 2]
    assert trajectories[0].states[0].origin == "Scrum"
    assert trajectories[0].outcome == "For Converted Try (+7)"
