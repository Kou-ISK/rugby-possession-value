# Rugby Possession Value

This package estimates rugby-union state value as **Expected Possession Value (EPV)** and values observed state transitions as **Expected Points Added (EPA)**.

The baseline is an absorbing Markov reward process. States combine possession origin, longitudinal zone, lateral zone and phase bucket. Transition probabilities use hierarchical empirical-Bayes smoothing, while uncertainty is estimated by match-cluster bootstrap.

The model estimates value under the historical observed policy. It does not claim counterfactual optimal action values from phase-only data.
