from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.linalg import solve

from .preprocess import Trajectory
from .schema import OUTCOME_REWARD, State


@dataclass(frozen=True)
class ModelConfig:
    alpha_absorb: float = 8.0
    alpha_transition: float = 12.0
    alpha_outcome: float = 8.0


class MarkovEPV:
    """Absorbing Markov reward process for observed-policy rugby EPV."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.states: list[State] = []
        self.outcomes: list[str] = list(OUTCOME_REWARD)
        self.q: NDArray[np.float64] | None = None
        self.b: NDArray[np.float64] | None = None
        self.absorption: NDArray[np.float64] | None = None
        self.values: NDArray[np.float64] | None = None
        self.state_visits: NDArray[np.int64] | None = None
        self.start_counts: NDArray[np.int64] | None = None

    def fit(self, trajectories: Iterable[Trajectory]) -> "MarkovEPV":
        episodes = list(trajectories)
        if not episodes:
            raise ValueError("At least one trajectory is required")
        self.states = sorted({state for episode in episodes for state in episode.states})
        state_index = {state: i for i, state in enumerate(self.states)}
        outcome_index = {outcome: i for i, outcome in enumerate(self.outcomes)}
        n, m = len(self.states), len(self.outcomes)

        continue_counts = np.zeros((n, n), dtype=float)
        terminal_counts = np.zeros((n, m), dtype=float)
        visits = np.zeros(n, dtype=np.int64)
        starts = np.zeros(n, dtype=np.int64)

        for episode in episodes:
            starts[state_index[episode.states[0]]] += 1
            for pos, state in enumerate(episode.states):
                i = state_index[state]
                visits[i] += 1
                if pos + 1 < len(episode.states):
                    continue_counts[i, state_index[episode.states[pos + 1]]] += 1
                else:
                    terminal_counts[i, outcome_index[episode.outcome]] += 1

        global_absorb = terminal_counts.sum() / (terminal_counts.sum() + continue_counts.sum())
        global_cont = _normalise(continue_counts.sum(axis=0))
        global_outcome = _normalise(terminal_counts.sum(axis=0))

        q = np.zeros((n, n), dtype=float)
        b = np.zeros((n, m), dtype=float)

        for i, state in enumerate(self.states):
            cont_n = continue_counts[i].sum()
            term_n = terminal_counts[i].sum()
            total_n = cont_n + term_n
            p_absorb = (term_n + self.config.alpha_absorb * global_absorb) / (
                total_n + self.config.alpha_absorb
            )

            peer_indices = [
                j for j, peer in enumerate(self.states)
                if peer.origin == state.origin and peer.phase_bucket == state.phase_bucket
            ]
            peer_cont = continue_counts[peer_indices].sum(axis=0) if peer_indices else np.zeros(n)
            transition_prior = _normalise(peer_cont, fallback=global_cont)

            peer_terminal = terminal_counts[peer_indices].sum(axis=0) if peer_indices else np.zeros(m)
            outcome_prior = _normalise(peer_terminal, fallback=global_outcome)

            conditional_cont = (continue_counts[i] + self.config.alpha_transition * transition_prior) / (
                cont_n + self.config.alpha_transition
            )
            conditional_outcome = (
                terminal_counts[i] + self.config.alpha_outcome * outcome_prior
            ) / (term_n + self.config.alpha_outcome)

            q[i] = (1.0 - p_absorb) * conditional_cont
            b[i] = p_absorb * conditional_outcome

        identity = np.eye(n)
        fundamental_absorption = solve(identity - q, b, assume_a="gen")
        rewards = np.array([OUTCOME_REWARD[o] for o in self.outcomes], dtype=float)
        values = fundamental_absorption @ rewards

        self.q, self.b = q, b
        self.absorption = fundamental_absorption
        self.values = values
        self.state_visits = visits
        self.start_counts = starts
        self._validate()
        return self

    def value(self, state: State) -> float:
        return float(self._require_values()[self.states.index(state)])

    def outcome_probabilities(self, state: State) -> dict[str, float]:
        row = self._require_absorption()[self.states.index(state)]
        return {outcome: float(row[i]) for i, outcome in enumerate(self.outcomes)}

    def state_frame(self) -> pd.DataFrame:
        values = self._require_values()
        absorption = self._require_absorption()
        visits = self._require_visits()
        starts = self._require_starts()
        rows: list[dict[str, object]] = []
        for i, state in enumerate(self.states):
            probs = {f"p_{_slug(o)}": float(absorption[i, j]) for j, o in enumerate(self.outcomes)}
            row: dict[str, object] = {
                "state_key": state.key,
                "origin": state.origin,
                "location": state.location,
                "side": state.side,
                "phase_bucket": state.phase_bucket,
                "x": state.x,
                "y": state.y,
                "epv": float(values[i]),
                "state_visits": int(visits[i]),
                "start_count": int(starts[i]),
                "p_score_for": float(sum(absorption[i, j] for j, o in enumerate(self.outcomes) if o.startswith("For "))),
                "p_try_for": float(sum(absorption[i, j] for j, o in enumerate(self.outcomes) if "Try" in o and o.startswith("For "))),
                "p_score_against": float(sum(absorption[i, j] for j, o in enumerate(self.outcomes) if o.startswith("Against "))),
                "p_no_score": float(absorption[i, self.outcomes.index("End of Half / Match (0)")]),
            }
            row.update(probs)
            rows.append(row)
        return pd.DataFrame(rows)

    def transition_frame(self, minimum_probability: float = 0.0) -> pd.DataFrame:
        q, b = self._require_q(), self._require_b()
        rows: list[dict[str, object]] = []
        for i, source in enumerate(self.states):
            for j, target in enumerate(self.states):
                probability = float(q[i, j])
                if probability >= minimum_probability:
                    rows.append({
                        "source_state": source.key,
                        "transition_type": "continue",
                        "target": target.key,
                        "probability": probability,
                        "target_epv": float(self._require_values()[j]),
                        "value_contribution": probability * float(self._require_values()[j]),
                    })
            for j, outcome in enumerate(self.outcomes):
                probability = float(b[i, j])
                if probability >= minimum_probability:
                    reward = OUTCOME_REWARD[outcome]
                    rows.append({
                        "source_state": source.key,
                        "transition_type": "absorb",
                        "target": outcome,
                        "probability": probability,
                        "target_epv": reward,
                        "value_contribution": probability * reward,
                    })
        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": "Rugby Possession Value",
            "model_type": "absorbing_markov_reward_process",
            "version": "0.1.0",
            "config": asdict(self.config),
            "states": [state.key for state in self.states],
            "outcomes": self.outcomes,
            "rewards": OUTCOME_REWARD,
            "q": self._require_q().tolist(),
            "b": self._require_b().tolist(),
            "absorption": self._require_absorption().tolist(),
            "values": self._require_values().tolist(),
            "state_visits": self._require_visits().tolist(),
            "start_counts": self._require_starts().tolist(),
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MarkovEPV":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(ModelConfig(**payload["config"]))
        model.states = [State.from_key(value) for value in payload["states"]]
        model.outcomes = list(payload["outcomes"])
        model.q = np.asarray(payload["q"], dtype=float)
        model.b = np.asarray(payload["b"], dtype=float)
        model.absorption = np.asarray(payload["absorption"], dtype=float)
        model.values = np.asarray(payload["values"], dtype=float)
        model.state_visits = np.asarray(payload["state_visits"], dtype=np.int64)
        model.start_counts = np.asarray(payload["start_counts"], dtype=np.int64)
        model._validate()
        return model

    def _validate(self) -> None:
        q, b = self._require_q(), self._require_b()
        row_sums = q.sum(axis=1) + b.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-8):
            raise ValueError("Transition rows do not sum to one")
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(q))))
        if spectral_radius >= 1.0:
            raise ValueError(f"Transient matrix is not absorbing: radius={spectral_radius}")
        rewards = np.array([OUTCOME_REWARD[o] for o in self.outcomes])
        residual = self._require_values() - (q @ self._require_values() + b @ rewards)
        if float(np.max(np.abs(residual))) > 1e-7:
            raise ValueError("Bellman residual exceeds tolerance")

    def _require_q(self) -> NDArray[np.float64]:
        if self.q is None:
            raise RuntimeError("Model is not fitted")
        return self.q

    def _require_b(self) -> NDArray[np.float64]:
        if self.b is None:
            raise RuntimeError("Model is not fitted")
        return self.b

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

    def _require_starts(self) -> NDArray[np.int64]:
        if self.start_counts is None:
            raise RuntimeError("Model is not fitted")
        return self.start_counts


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


def _slug(value: str) -> str:
    return (
        value.lower().replace(" / ", "_").replace(" ", "_").replace("(", "")
        .replace(")", "").replace("+", "plus").replace("-", "minus")
    )
