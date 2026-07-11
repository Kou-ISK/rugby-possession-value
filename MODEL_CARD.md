# Model Card

## Intended use

Descriptive EPV analysis of rugby-union possessions and observed EPA analysis of phase transitions.

## Not intended for

- causal claims about action choice;
- live safety-critical decisions;
- direct comparison of players without role, team and opportunity adjustment;
- claiming optimal keep/kick decisions from phase-only data.

## Training data

2018/19 English Premiership Rugby phase data, fetched from the public upstream repository.

## Estimator

Absorbing Markov reward process with empirical-Bayes smoothing and match-cluster bootstrap.

## Known limitations

Coarse spatial zones, approximate coordinates, no action-level pass/carry/kick labels, no player identifiers, historical-policy dependence, and remaining team/opponent/context confounding.
