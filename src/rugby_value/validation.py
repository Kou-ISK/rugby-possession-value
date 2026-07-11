from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import MarkovEPV, ModelConfig
from .preprocess import Trajectory
from .schema import OUTCOME_REWARD


@dataclass(frozen=True)
class CrossValidationConfig:
    folds: int = 5
    seed: int = 20260711


def match_group_cross_validate(
    trajectories: list[Trajectory],
    model_config: ModelConfig,
    config: CrossValidationConfig = CrossValidationConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate start-state EPV and terminal-outcome probabilities by match folds."""
    match_ids = np.array(sorted({trajectory.match_id for trajectory in trajectories}))
    if len(match_ids) < config.folds:
        raise ValueError("Number of matches must be at least the number of folds")
    rng = np.random.default_rng(config.seed)
    shuffled = rng.permutation(match_ids)
    fold_ids = np.array_split(shuffled, config.folds)
    rows: list[dict[str, object]] = []

    for fold, test_ids in enumerate(fold_ids):
        test_set = set(str(value) for value in test_ids)
        train = [t for t in trajectories if t.match_id not in test_set]
        test = [t for t in trajectories if t.match_id in test_set]
        model = MarkovEPV(model_config).fit(train)
        known_states = set(model.states)
        for trajectory in test:
            start = trajectory.states[0]
            if start not in known_states:
                continue
            probabilities = model.outcome_probabilities(start)
            actual = OUTCOME_REWARD[trajectory.outcome]
            predicted = model.value(start)
            outcome_probability = max(probabilities.get(trajectory.outcome, 0.0), 1e-15)
            rows.append({
                "fold": fold,
                "match_id": trajectory.match_id,
                "possession_id": trajectory.possession_id,
                "start_state": start.key,
                "actual_points": actual,
                "predicted_epv": predicted,
                "error": actual - predicted,
                "squared_error": (actual - predicted) ** 2,
                "negative_log_likelihood": -float(np.log(outcome_probability)),
            })

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise ValueError("No test start states were observed in training folds")
    summary = pd.DataFrame([{
        "n": len(predictions),
        "coverage": len(predictions) / len(trajectories),
        "mae": float(predictions["error"].abs().mean()),
        "rmse": float(np.sqrt(predictions["squared_error"].mean())),
        "mean_error": float(predictions["error"].mean()),
        "multiclass_log_loss": float(predictions["negative_log_likelihood"].mean()),
    }])
    return summary, predictions
