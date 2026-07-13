# Rugby Possession Value

[日本語](docs/ja/README.md) | [English](docs/en/README.md)

An open-source, reproducible framework for estimating **Expected Possession Value (EPV)** and **Expected Points Added (EPA)** in rugby union.

Version 0.2 models the game **across possession changes**. A kick or turnover does not terminate value estimation: the chain continues into the opponent's next state, with that state's value sign-flipped back into the source team's perspective.

## Core recursion

```text
V(s)
= Σ P_same(s→s') V(s')
+ Σ P_opp(s→s') [-V(s')]
+ Σ P_score(s→o) u(o)
```

The source dataset is ordered by match and phase ID. Consecutive rows are used to estimate:

- same-team state transitions;
- opponent-possession state transitions;
- scoring-event absorption.

A scoring event between two rows is detected by converting the next row's `Points_Difference` back to the source team's perspective.

## Quick start

```bash
python -m pip install -e ".[dev]"
rugby-value fetch-data
rugby-value fit data/raw/phase_2018-19.csv --output models/premiership-2018-19
rugby-value table models/premiership-2018-19/model.json --output data/processed
```

Outputs include:

- `state_values.csv`
- `start_state_values.csv`
- `transition_probabilities.csv`
- `student_table.csv`
- `observed_epa.csv`

## Interpretation

This is an **observed-policy state-value model**. It correctly carries value through possession changes, but the public phase data does not label individual choices such as carry, pass, box kick or territory kick. Those counterfactual action values require action-labelled event data.

## References

- Sawczuk, Palczewska & Jones (2021), *Development of an expected possession value model to analyse team attacking performances in rugby league*.
- Sawczuk et al. (2022), *A Bayesian Mixture Model Approach to Expected Possession Values in Rugby League*.
- Fernández, Bornn & Cervone (2020), *A framework for the fine-grained evaluation of the instantaneous expected value of soccer possessions*.
- Decroos et al. (2019), *Actions Speak Louder Than Goals: Valuing Player Actions in Soccer*.
- Martinez-Arastey et al. (2025), *Foundations of expected points in rugby union: A methodological approach*.

## License

Apache-2.0. Source dataset licensing remains governed by its upstream source.
