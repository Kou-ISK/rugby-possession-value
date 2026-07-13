from pathlib import Path

from rugby_value.preprocess import build_steps, load_phase_data, prepare_observations

FIXTURE = Path(__file__).parent / "fixtures" / "phases.csv"


def test_build_steps_detects_possession_change_and_score() -> None:
    observations = prepare_observations(load_phase_data(FIXTURE))
    steps = build_steps(observations)
    assert len(steps) == len(observations)
    assert steps[0].target_state is not None
    assert steps[0].possession_flipped is True
    assert steps[2].absorb_outcome == "For Penalty Kick (+3)"
    assert steps[-1].absorb_outcome == "End of Half / Match (0)"
