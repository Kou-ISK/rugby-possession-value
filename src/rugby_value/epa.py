from __future__ import annotations

import pandas as pd

from .model import CrossPossessionEPV
from .preprocess import Step
from .schema import OUTCOME_REWARD


def observed_epa(model: CrossPossessionEPV, steps: list[Step]) -> pd.DataFrame:
    """Calculate EPA for observed same-team, opponent-team and scoring transitions."""
    values = {state: model.value(state) for state in model.states}
    rows: list[dict[str, object]] = []
    for step in steps:
        before = values[step.source_state]
        if step.absorb_outcome is not None:
            after = 0.0
            reward = OUTCOME_REWARD[step.absorb_outcome]
            transition_type = "absorbing_outcome"
            target = step.absorb_outcome
        elif step.target_state is not None:
            target_value = values[step.target_state]
            after = -target_value if step.possession_flipped else target_value
            reward = 0.0
            transition_type = (
                "opponent_possession" if step.possession_flipped else "same_possession"
            )
            target = step.target_state.key
        else:
            raise ValueError("A step must have either target_state or absorb_outcome")
        rows.append(
            {
                "source_state": step.source_state.key,
                "transition_type": transition_type,
                "target": target,
                "epv_before": before,
                "epv_after_from_source_perspective": after,
                "immediate_reward": reward,
                "epa": reward + after - before,
            }
        )
    return pd.DataFrame(rows)
