# Methodology

The estimator reconstructs possessions from phase sequences and treats each possession as a trajectory through transient states to an absorbing scoring outcome. Only the final phase contributes an absorbing transition, preventing long possessions from being over-counted.

State:

```text
(origin, location_zone, lateral_zone, phase_bucket)
```

The model uses a Beta-Binomial empirical-Bayes estimator for absorption probability, Dirichlet-Multinomial estimators for continuation and terminal distributions, and a match-cluster bootstrap for uncertainty.

```text
Q[s,s'] = (1-p_absorb(s)) P(s'|continue,s)
B[s,o]  = p_absorb(s) P(o|absorb,s)
H       = (I-Q)^(-1) B
V       = H u
```

The resulting EPV is descriptive under the observed policy, not causal or optimal.
