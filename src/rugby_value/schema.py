from __future__ import annotations

from dataclasses import dataclass

LOCATION_TO_X: dict[str, float] = {
    "Goal-5m (own)": 2.5,
    "5m-22m (own)": 13.5,
    "22m-10m (own)": 31.0,
    "10m-Half (own)": 45.0,
    "Half-10m (opp)": 55.0,
    "10m-22m (opp)": 69.0,
    "22m-5m (opp)": 86.5,
    "5m-Goal (opp)": 97.5,
}
SIDE_TO_Y: dict[str, float] = {"Left": 70 / 6, "Centre": 35.0, "Right": 70 * 5 / 6}

OUTCOME_REWARD: dict[str, float] = {
    "For Converted Try (+7)": 7.0,
    "For Try (+5)": 5.0,
    "For Penalty Kick (+3)": 3.0,
    "For Drop Goal (+3)": 3.0,
    "Against Converted Try (-7)": -7.0,
    "Against Try (-5)": -5.0,
    "Against Penalty Kick (-3)": -3.0,
    "Against Drop Goal (-3)": -3.0,
    "End of Half / Match (0)": 0.0,
}
MIRROR_OUTCOME: dict[str, str] = {
    "For Converted Try (+7)": "Against Converted Try (-7)",
    "For Try (+5)": "Against Try (-5)",
    "For Penalty Kick (+3)": "Against Penalty Kick (-3)",
    "For Drop Goal (+3)": "Against Drop Goal (-3)",
    "Against Converted Try (-7)": "For Converted Try (+7)",
    "Against Try (-5)": "For Try (+5)",
    "Against Penalty Kick (-3)": "For Penalty Kick (+3)",
    "Against Drop Goal (-3)": "For Drop Goal (+3)",
    "End of Half / Match (0)": "End of Half / Match (0)",
}


def phase_bucket(phase: int) -> str:
    if phase <= 1:
        return "1"
    if phase <= 3:
        return "2-3"
    if phase <= 6:
        return "4-6"
    return "7+"


@dataclass(frozen=True, order=True)
class State:
    origin: str
    location: str
    side: str
    phase_bucket: str

    @property
    def key(self) -> str:
        return "|".join((self.origin, self.location, self.side, self.phase_bucket))

    @property
    def x(self) -> float:
        return LOCATION_TO_X[self.location]

    @property
    def y(self) -> float:
        return SIDE_TO_Y[self.side]

    @classmethod
    def from_key(cls, value: str) -> "State":
        origin, location, side, bucket = value.split("|", maxsplit=3)
        return cls(origin, location, side, bucket)
