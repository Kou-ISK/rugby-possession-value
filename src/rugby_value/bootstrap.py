from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import MarkovEPV, ModelConfig
from .preprocess import Trajectory


@dataclass(frozen=True)
class BootstrapConfig:
    iterations: int = 500
    seed: int = 20260711
    lower: float = 0.05
    upper: float = 0.95


def cluster_bootstrap(
    trajectories: list[Trajectory],
    model_config: ModelConfig,
    config: BootstrapConfig,
) -> pd.DataFrame:
    """Match-cluster bootstrap intervals for EPV."""
    by_match: dict[str, list[Trajectory]] = defaultdict(list)
    for trajectory in trajectories:
        by_match[trajectory.match_id].append(trajectory)
    match_ids = np.array(sorted(by_match))
    rng = np.random.default_rng(config.seed)

    reference = MarkovEPV(model_config).fit(trajectories)
    state_keys = [state.key for state in reference.states]
    samples = np.full((config.iterations, len(state_keys)), np.nan)

    for iteration in range(config.iterations):
        selected = rng.choice(match_ids, size=len(match_ids), replace=True)
        sampled: list[Trajectory] = []
        for sample_number, match_id in enumerate(selected):
            for trajectory in by_match[str(match_id)]:
                sampled.append(Trajectory(
                    match_id=f"boot-{sample_number}-{trajectory.match_id}",
                    possession_id=f"boot-{sample_number}-{trajectory.possession_id}",
                    states=trajectory.states,
                    outcome=trajectory.outcome,
                ))
        fitted = MarkovEPV(model_config).fit(sampled)
        if fitted.values is None:
            raise RuntimeError("Bootstrap model did not produce values")
        lookup = dict(zip((s.key for s in fitted.states), fitted.values, strict=True))
        samples[iteration] = [lookup.get(key, np.nan) for key in state_keys]

    return pd.DataFrame({
        "state_key": state_keys,
        "epv_p05": np.nanquantile(samples, config.lower, axis=0),
        "epv_p50": np.nanquantile(samples, 0.5, axis=0),
        "epv_p95": np.nanquantile(samples, config.upper, axis=0),
        "bootstrap_valid_n": np.sum(~np.isnan(samples), axis=0),
    })
