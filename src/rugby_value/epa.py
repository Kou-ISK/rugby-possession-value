from __future__ import annotations

import pandas as pd

from .model import MarkovEPV
from .preprocess import Trajectory
from .schema import OUTCOME_REWARD


def observed_epa(model: MarkovEPV, trajectories: list[Trajectory]) -> pd.DataFrame:
    """Calculate EPA for observed phase-to-phase and terminal transitions."""
    values = {state: model.value(state) for state in model.states}
    rows: list[dict[str, object]] = []
    for trajectory in trajectories:
        for index, state in enumerate(trajectory.states):
            before = values[state]
            if index + 1 < len(trajectory.states):
                after_state = trajectory.states[index + 1]
                after = values[after_state]
                reward = 0.0
                transition_type = "continue"
                target = after_state.key
            else:
                after = 0.0
                reward = OUTCOME_REWARD[trajectory.outcome]
                transition_type = "absorb"
                target = trajectory.outcome
            rows.append({
                "match_id": trajectory.match_id,
                "possession_id": trajectory.possession_id,
                "phase_index": index + 1,
                "source_state": state.key,
                "transition_type": transition_type,
                "target": target,
                "epv_before": before,
                "epv_after": after,
                "immediate_reward": reward,
                "epa": reward + after - before,
            })
    return pd.DataFrame(rows)
