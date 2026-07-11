from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .schema import OUTCOME_REWARD, State, phase_bucket

REQUIRED_COLUMNS = {
    "ID", "Round", "Home", "Away", "Phase", "Team_In_Poss", "Location", "Side",
    "Play_Start", "Points_Difference", "Seconds_Remaining", "Outcome",
}


@dataclass(frozen=True)
class Trajectory:
    match_id: str
    possession_id: str
    states: tuple[State, ...]
    outcome: str


def load_phase_data(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    unknown = set(frame["Outcome"].dropna().unique()) - set(OUTCOME_REWARD)
    if unknown:
        raise ValueError(f"Unknown outcomes: {sorted(unknown)}")
    return frame


def prepare_trajectories(frame: pd.DataFrame) -> list[Trajectory]:
    """Reconstruct possession trajectories from ordered phase rows.

    A new possession starts on Phase == 1. Defensive fallbacks detect team changes,
    non-increasing IDs, and phase-number resets in imperfect input data.
    """
    df = frame.copy()
    df["match_id"] = (
        df["Round"].astype(str) + "|" + df["Home"].astype(str) + "|" + df["Away"].astype(str)
    )
    df = df.sort_values(["match_id", "ID"], kind="stable").reset_index(drop=True)

    trajectories: list[Trajectory] = []
    for match_id, match in df.groupby("match_id", sort=False):
        current: list[pd.Series] = []
        run_number = 0
        previous: pd.Series | None = None
        for _, row in match.iterrows():
            starts_new = previous is None or int(row["Phase"]) == 1
            if previous is not None:
                starts_new = starts_new or row["Team_In_Poss"] != previous["Team_In_Poss"]
                starts_new = starts_new or int(row["Phase"]) <= int(previous["Phase"])
                starts_new = starts_new or int(row["ID"]) <= int(previous["ID"])
            if starts_new and current:
                trajectories.append(_to_trajectory(match_id, run_number, current))
                current = []
                run_number += 1
            current.append(row)
            previous = row
        if current:
            trajectories.append(_to_trajectory(match_id, run_number, current))
    return trajectories


def _to_trajectory(match_id: str, run_number: int, rows: list[pd.Series]) -> Trajectory:
    origin = str(rows[0]["Play_Start"])
    outcome = str(rows[-1]["Outcome"])
    # The published data normally repeats the possession outcome across phases.
    # The final row is authoritative; inconsistencies are deliberately not duplicated.
    states = tuple(
        State(
            origin=origin,
            location=str(row["Location"]),
            side=str(row["Side"]),
            phase_bucket=phase_bucket(int(row["Phase"])),
        )
        for row in rows
    )
    return Trajectory(
        match_id=match_id,
        possession_id=f"{match_id}|{run_number}",
        states=states,
        outcome=outcome,
    )
