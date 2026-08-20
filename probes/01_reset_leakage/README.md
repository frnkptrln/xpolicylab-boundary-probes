# 01 — Reset leakage

## Hypothesis

After `reset()`, a target observation produces the same action as the same
observation on a fresh model instance, even after a different priming episode.

## Comparison

```text
reused: reset → prime → act → reset → target → act
fresh:  reset → target → act
```

The two target actions must match at absolute tolerance `1e-7`. A deliberately
broken stateful canary is included: the probe must reject it or the detector
itself fails.

## Interpretation

A pass demonstrates detector sensitivity and, when applied to a target
adapter, absence of action-visible leakage for the chosen prime/target pair.
It cannot prove that all internal state was cleared.
