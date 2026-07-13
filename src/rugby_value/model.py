from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.linalg import solve

from .preprocess import Step
from .schema import MIRROR_OUTCOME, OUTCOME_REWARD, State


@dataclass(frozen=True)
class ModelConfig:
    alpha_same: float = 12.0
    alpha_opp: float = 12.0
    alpha_absorb: float = 8.0


class CrossPossessionEPV:
    """Signed Markov reward process spanning possession changes.

    Values are always expressed from the source state's possession-team perspective.
    A transition to an opponent-possession state therefore contributes ``-V(s')``.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.states: list[State] = []
        self.outcomes: list[str] = list(OUTCOME_REWARD)
        self.p_same: NDArray[np.float64] | None = None
        self.p_opp: NDArray[np.float64] | None = None
        self.p_absorb: NDArray[np.float64] | None = None
        self.absorption: NDArray[np.float64] | None = None
        self.values: NDArray[np.float64] | None = None
        self.state_visits: NDArray[np.int64] | None = None

    def fit(self, steps: Iterable[Step]) -> "CrossPossessionEPV":
        observations = list(steps)
        if not observations:
            raise ValueError("At least one transition step is required")

        self.states = sorted(
            {step.source_state for step in observations}
            | {step.target_state for step in observations if step.target_state is not None}
        )
        state_index = {state: i for i, state in enumerate(self.states)}
        outcome_index = {outcome: i for i, outcome in enumerate(self.outcomes)}
        n, m = len(self.states), len(self.outcomes)

        same_counts = np.zeros((n, n), dtype=float)
        opp_counts = np.zeros((n, n), dtype=float)
        absorb_counts = np.zeros((n, m), dtype=float)
        visits = np.zeros(n, dtype=np.int64)

        for step in observations:
            i = state_index[step.source_state]
            visits[i] += 1
            if step.absorb_outcome is not None:
                absorb_counts[i, outcome_index[step.absorb_outcome]] += 1
            elif step.target_state is not None:
                j = state_index[step.target_state]
                matrix = opp_counts if step.possession_flipped else same_counts
                matrix[i, j] += 1
            else:
                raise ValueError("A step must have either target_state or absorb_outcome")

        global_same = _normalise(same_counts.sum(axis=0))
        global_opp = _normalise(opp_counts.sum(axis=0))
        global_absorb = _normalise(absorb_counts.sum(axis=0))
        global_split = _normalise(
            np.array([same_counts.sum(), opp_counts.sum(), absorb_counts.sum()], dtype=float)
        )

        p_same = np.zeros_like(same_counts)
        p_opp = np.zeros_like(opp_counts)
        p_absorb = np.zeros_like(absorb_counts)

        for i, state in enumerate(self.states):
            peers = [
                j
                for j, peer in enumerate(self.states)
                if peer.origin == state.origin and peer.phase_bucket == state.phase_bucket
            ]
            same_prior = _normalise(same_counts[peers].sum(axis=0), global_same)
            opp_prior = _normalise(opp_counts[peers].sum(axis=0), global_opp)
            absorb_prior = _normalise(absorb_counts[peers].sum(axis=0), global_absorb)

            same_n = same_counts[i].sum()
            opp_n = opp_counts[i].sum()
            absorb_n = absorb_counts[i].sum()
            total_n = same_n + opp_n + absorb_n
            split = (
                np.array([same_n, opp_n, absorb_n], dtype=float)
                + self.config.alpha_absorb * global_split
            ) / (total_n + self.config.alpha_absorb)

            same_cond = (
                same_counts[i] + self.config.alpha_same * same_prior
            ) / (same_n + self.config.alpha_same)
            opp_cond = (
                opp_counts[i] + self.config.alpha_opp * opp_prior
            ) / (opp_n + self.config.alpha_opp)
            absorb_cond = (
                absorb_counts[i] + self.config.alpha_absorb * absorb_prior
            ) / (absorb_n + self.config.alpha_absorb)

            p_same[i] = split[0] * same_cond
            p_opp[i] = split[1] * opp_cond
            p_absorb[i] = split[2] * absorb_cond

        absorption = self._solve_outcome_probabilities(p_same, p_opp, p_absorb)
        rewards = np.array([OUTCOME_REWARD[o] for o in self.outcomes], dtype=float)
        values = absorption @ rewards

        self.p_same = p_same
        self.p_opp = p_opp
        self.p_absorb = p_absorb
        self.absorption = absorption
        self.values = values
        self.state_visits = visits
        self._validate()
        return self

    def _solve_outcome_probabilities(
        self,
        p_same: NDArray[np.float64],
        p_opp: NDArray[np.float64],
        p_absorb: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        n, m = p_absorb.shape
        mirror = np.array(
            [self.outcomes.index(MIRROR_OUTCOME[o]) for o in self.outcomes], dtype=int
        )
        system = np.zeros((n * m, n * m), dtype=float)
        rhs = p_absorb.T.reshape(n * m)
        identity = np.eye(n)

        for outcome_i in range(m):
            row = slice(outcome_i * n, (outcome_i + 1) * n)
            system[row, row] = identity - p_same
            mirror_col = slice(mirror[outcome_i] * n, (mirror[outcome_i] + 1) * n)
            system[row, mirror_col] -= p_opp

        return solve(system, rhs, assume_a="gen").reshape(m, n).T

    def value(self, state: State) -> float:
        return float(self._require_values()[self.states.index(state)])

    def outcome_probabilities(self, state: State) -> dict[str, float]:
        row = self._require_absorption()[self.states.index(state)]
        return {outcome: float(row[i]) for i, outcome in enumerate(self.outcomes)}

    def state_frame(self) -> pd.DataFrame:
        values = self._require_values()
        absorption = self._require_absorption()
        visits = self._require_visits()
        rows: list[dict[str, object]] = []
        for i, state in enumerate(self.states):
            rows.append(
                {
                    "state_key": state.key,
                    "origin": state.origin,
                    "location": state.location,
                    "side": state.side,
                    "phase_bucket": state.phase_bucket,
                    "x": state.x,
                    "y": state.y,
                    "epv": float(values[i]),
                    "state_visits": int(visits[i]),
                    "p_score_for": float(
                        sum(absorption[i, j] for j, o in enumerate(self.outcomes) if o.startswith("For "))
                    ),
                    "p_try_for": float(
                        sum(
                            absorption[i, j]
                            for j, o in enumerate(self.outcomes)
                            if o.startswith("For ") and "Try" in o
                        )
                    ),
                    "p_score_against": float(
                        sum(
                            absorption[i, j]
                            for j, o in enumerate(self.outcomes)
                            if o.startswith("Against ")
                        )
                    ),
                    "p_no_score": float(
                        absorption[i, self.outcomes.index("End of Half / Match (0)")]
                    ),
                }
            )
        return pd.DataFrame(rows)

    def transition_frame(self, minimum_probability: float = 0.0) -> pd.DataFrame:
        p_same = self._require_same()
        p_opp = self._require_opp()
        p_absorb = self._require_absorb()
        values = self._require_values()
        rows: list[dict[str, object]] = []
        for i, source in enumerate(self.states):
            for j, target in enumerate(self.states):
                same_probability = float(p_same[i, j])
                if same_probability >= minimum_probability:
                    rows.append(
                        {
                            "source_state": source.key,
                            "transition_type": "same_possession",
                            "target": target.key,
                            "probability": same_probability,
                            "target_epv_from_source_perspective": float(values[j]),
                            "value_contribution": same_probability * float(values[j]),
                        }
                    )
                opp_probability = float(p_opp[i, j])
                if opp_probability >= minimum_probability:
                    rows.append(
                        {
                            "source_state": source.key,
                            "transition_type": "opponent_possession",
                            "target": target.key,
                            "probability": opp_probability,
                            "target_epv_from_source_perspective": float(-values[j]),
                            "value_contribution": opp_probability * float(-values[j]),
                        }
                    )
            for j, outcome in enumerate(self.outcomes):
                probability = float(p_absorb[i, j])
                if probability >= minimum_probability:
                    reward = OUTCOME_REWARD[outcome]
                    rows.append(
                        {
                            "source_state": source.key,
                            "transition_type": "absorbing_outcome",
                            "target": outcome,
                            "probability": probability,
                            "target_epv_from_source_perspective": reward,
                            "value_contribution": probability * reward,
                        }
                    )
        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> None:
        payload = {
            "model_name": "Rugby Possession Value",
            "model_type": "cross_possession_signed_markov_reward_process",
            "version": "0.2.0",
            "config": asdict(self.config),
            "states": [state.key for state in self.states],
            "outcomes": self.outcomes,
            "p_same": self._require_same().tolist(),
            "p_opp": self._require_opp().tolist(),
            "p_absorb": self._require_absorb().tolist(),
            "absorption": self._require_absorption().tolist(),
            "values": self._require_values().tolist(),
            "state_visits": self._require_visits().tolist(),
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CrossPossessionEPV":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(ModelConfig(**payload["config"]))
        model.states = [State.from_key(value) for value in payload["states"]]
        model.outcomes = list(payload["outcomes"])
        model.p_same = np.asarray(payload["p_same"], dtype=float)
        model.p_opp = np.asarray(payload["p_opp"], dtype=float)
        model.p_absorb = np.asarray(payload["p_absorb"], dtype=float)
        model.absorption = np.asarray(payload["absorption"], dtype=float)
        model.values = np.asarray(payload["values"], dtype=float)
        model.state_visits = np.asarray(payload["state_visits"], dtype=np.int64)
        model._validate()
        return model

    def _validate(self) -> None:
        row_sums = (
            self._require_same().sum(axis=1)
            + self._require_opp().sum(axis=1)
            + self._require_absorb().sum(axis=1)
        )
        if not np.allclose(row_sums, 1.0, atol=1e-8):
            raise ValueError("Transition rows do not sum to one")
        rewards = np.array([OUTCOME_REWARD[o] for o in self.outcomes], dtype=float)
        values = self._require_values()
        residual = values - (
            self._require_same() @ values
            + self._require_opp() @ (-values)
            + self._require_absorb() @ rewards
        )
        if not np.allclose(residual, 0.0, atol=1e-7):
            raise ValueError("Bellman residual exceeds tolerance")
        if not np.allclose(self._require_absorption().sum(axis=1), 1.0, atol=1e-7):
            raise ValueError("Outcome probabilities do not sum to one")

    def _require_same(self) -> NDArray[np.float64]:
        if self.p_same is None:
            raise RuntimeError("Model is not fitted")
        return self.p_same

    def _require_opp(self) -> NDArray[np.float64]:
        if self.p_opp is None:
            raise RuntimeError("Model is not fitted")
        return self.p_opp

    def _require_absorb(self) -> NDArray[np.float64]:
        if self.p_absorb is None:
            raise RuntimeError("Model is not fitted")
        return self.p_absorb

    def _require_absorption(self) -> NDArray[np.float64]:
        if self.absorption is None:
            raise RuntimeError("Model is not fitted")
        return self.absorption

    def _require_values(self) -> NDArray[np.float64]:
        if self.values is None:
            raise RuntimeError("Model is not fitted")
        return self.values

    def _require_visits(self) -> NDArray[np.int64]:
        if self.state_visits is None:
            raise RuntimeError("Model is not fitted")
        return self.state_visits


def _normalise(
    values: NDArray[np.float64],
    fallback: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    total = float(values.sum())
    if total > 0:
        return values / total
    if fallback is not None:
        return fallback.copy()
    return np.full(values.shape, 1.0 / len(values), dtype=float)


MarkovEPV = CrossPossessionEPV
