from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .schema import OUTCOME_REWARD, State, phase_bucket

REQUIRED_COLUMNS = {
    "ID",
    "Round",
    "Home",
    "Away",
    "Phase",
    "Team_In_Poss",
    "Location",
    "Side",
    "Play_Start",
    "Points_Difference",
    "Seconds_Remaining",
    "Outcome",
}


@dataclass(frozen=True)
class Observation:
    match_id: str
    row_id: int
    team_in_poss: str
    points_difference: float
    outcome: str
    state: State


@dataclass(frozen=True)
class Step:
    source_state: State
    target_state: State | None
    possession_flipped: bool
    absorb_outcome: str | None


def load_phase_data(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    unknown = set(frame["Outcome"].dropna().unique()) - set(OUTCOME_REWARD)
    if unknown:
        raise ValueError(f"Unknown outcomes: {sorted(unknown)}")
    return frame


def prepare_observations(frame: pd.DataFrame) -> list[Observation]:
    """Convert phase rows into one ordered game-state sequence per match."""
    data = frame.copy()
    data["match_id"] = (
        data["Round"].astype(str)
        + "|"
        + data["Home"].astype(str)
        + "|"
        + data["Away"].astype(str)
    )
    data = data.sort_values(["match_id", "ID"], kind="stable").reset_index(drop=True)

    return [
        Observation(
            match_id=str(row["match_id"]),
            row_id=int(row["ID"]),
            team_in_poss=str(row["Team_In_Poss"]),
            points_difference=float(row["Points_Difference"]),
            outcome=str(row["Outcome"]),
            state=State(
                origin=str(row["Play_Start"]),
                location=str(row["Location"]),
                side=str(row["Side"]),
                phase_bucket=phase_bucket(int(row["Phase"])),
            ),
        )
        for _, row in data.iterrows()
    ]


def build_steps(observations: list[Observation]) -> list[Step]:
    """Build signed transitions from consecutive rows.

    ``Points_Difference`` is expressed from the possession team's perspective.
    The next row is therefore converted back to the source team's perspective
    before checking whether a scoring event occurred between the rows.
    """
    steps: list[Step] = []
    for index, observation in enumerate(observations):
        is_match_end = (
            index == len(observations) - 1
            or observations[index + 1].match_id != observation.match_id
        )
        if is_match_end:
            steps.append(
                Step(
                    source_state=observation.state,
                    target_state=None,
                    possession_flipped=False,
                    absorb_outcome=observation.outcome,
                )
            )
            continue

        next_observation = observations[index + 1]
        possession_flipped = next_observation.team_in_poss != observation.team_in_poss
        next_difference_from_source = (
            -next_observation.points_difference
            if possession_flipped
            else next_observation.points_difference
        )
        score_delta = next_difference_from_source - observation.points_difference

        if abs(score_delta) > 1e-9:
            steps.append(
                Step(
                    source_state=observation.state,
                    target_state=None,
                    possession_flipped=possession_flipped,
                    absorb_outcome=observation.outcome,
                )
            )
        else:
            steps.append(
                Step(
                    source_state=observation.state,
                    target_state=next_observation.state,
                    possession_flipped=possession_flipped,
                    absorb_outcome=None,
                )
            )
    return steps


def prepare_steps(frame: pd.DataFrame) -> list[Step]:
    return build_steps(prepare_observations(frame))
