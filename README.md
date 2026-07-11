# Rugby Possession Value

[日本語](docs/ja/README.md) | [English](docs/en/README.md)

An open-source, reproducible framework for estimating **Expected Possession Value (EPV)** and **Expected Points Added (EPA)** in rugby union.

The project deliberately separates:

- **EPV**: the expected terminal net points of a game state under the observed policy;
- **EPA**: the change in EPV caused by an observed transition;
- **decision value**: counterfactual action comparison, which requires action-labelled data and is not inferred from phase-only data.

The baseline estimator is an **absorbing Markov reward process** with hierarchical empirical-Bayes smoothing and match-cluster bootstrap uncertainty.

## Quick start

```bash
python -m pip install -e ".[dev]"
rugby-value fetch-data
rugby-value fit data/raw/phase_2018-19.csv --output models/premiership-2018-19
rugby-value table models/premiership-2018-19/model.json --output data/processed
rugby-value validate data/raw/phase_2018-19.csv --output validation
```

## Core equation

For transient state transition `s -> s'`:

```text
EPA = EPV(s') - EPV(s)
```

For an absorbing transition with terminal reward `r`:

```text
EPA = r - EPV(s)
```

The EPV vector is obtained from:

```text
V = (I - Q)^(-1) B u
```

where `Q` is the transient transition matrix, `B` is the transition matrix to absorbing outcomes, and `u` is the outcome reward vector.

## Status

The repository provides a production-quality implementation and reproducible pipeline. The source dataset is not redistributed; the fetch command downloads it from its public upstream repository.

## References

- Sawczuk, Palczewska & Jones (2021), *Development of an expected possession value model to analyse team attacking performances in rugby league*.
- Sawczuk et al. (2022), *A Bayesian Mixture Model Approach to Expected Possession Values in Rugby League*.
- Fernández, Bornn & Cervone (2020), *A framework for the fine-grained evaluation of the instantaneous expected value of soccer possessions*.
- Decroos et al. (2019), *Actions Speak Louder Than Goals: Valuing Player Actions in Soccer*.
- Martinez-Arastey et al. (2025), *Foundations of expected points in rugby union: A methodological approach*.

## License

Apache-2.0. Source dataset licensing remains governed by its upstream source.
