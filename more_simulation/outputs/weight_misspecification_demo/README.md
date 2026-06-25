# Weight misspecification simulation

## Motivation

This simulation targets a potential weakness of weighted conformal selection under covariate shift:
the finite-sample FDR guarantee depends on using the correct density-ratio weights. The procedure is
model-free with respect to the outcome model, but it is not weight-free. If the covariate-shift weights
are ignored or strongly distorted, the weighted conformal p-values may no longer be calibrated for the
test distribution.

## Data-generating mechanism

- Calibration covariate: `X_cal ~ N(0, 1)`.
- Test covariate: `X_test ~ N(delta, 1)` with `delta=1.5`.
- Outcome: `Y | X ~ Bernoulli(sigmoid(beta0 + beta1 X))`, with `beta0=-1.0` and `beta1=2.0`.
- The test distribution is shifted toward larger X, where positives are more likely.
- We select test points believed to have `Y=1`; false discoveries are selected points with `Y=0`.

The true density ratio is:

```text
w(x) = dQ_X / dP_X = exp(delta * x - delta^2 / 2)
```

We compare three weighting strategies:

- oracle weights: use the true density ratio;
- ignore shift: set all weights to 1;
- clipped weights: use `min(w(x), 3.0)`.

## Why this probes the paper's limitation

Under covariate shift, test null examples are not distributed like calibration null examples. If high-X nulls
are underrepresented in calibration and we ignore weights, the conformal p-values can become too small in the
shifted region. This can increase the false discovery proportion. Correct weights repair the comparison by
reweighting calibration examples toward the test covariate distribution.

## Results

Nominal FDR level: `q=0.1`. Number of repetitions: `300`.

| weights | method | runs | mean_fdp | sd_fdp | prob_fdp_gt_q | mean_power | mean_nsel | mean_cal_positive_rate | mean_test_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle weights | WBH | 300 | 0.0981 | 0.0309 | 0.4867 | 0.8328 | 360.1633 | 0.3533 | 0.7761 |
| oracle weights | WCS.hete | 300 | 0.0995 | 0.0386 | 0.5000 | 0.6413 | 278.2200 | 0.3533 | 0.7761 |
| oracle weights | WCS.homo | 300 | 0.0940 | 0.0363 | 0.4800 | 0.7964 | 344.5067 | 0.3533 | 0.7761 |
| oracle weights | WCS.dtm | 300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3533 | 0.7761 |
| ignore shift | WBH | 300 | 0.1357 | 0.0181 | 0.9833 | 0.9438 | 423.8700 | 0.3533 | 0.7761 |
| ignore shift | WCS.hete | 300 | 0.1357 | 0.0182 | 0.9833 | 0.9437 | 423.8367 | 0.3533 | 0.7761 |
| ignore shift | WCS.homo | 300 | 0.1356 | 0.0181 | 0.9833 | 0.9437 | 423.7933 | 0.3533 | 0.7761 |
| ignore shift | WCS.dtm | 300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3533 | 0.7761 |
| clip weights at 3 | WBH | 300 | 0.0821 | 0.0211 | 0.1933 | 0.8131 | 344.2333 | 0.3533 | 0.7761 |
| clip weights at 3 | WCS.hete | 300 | 0.0822 | 0.0212 | 0.1967 | 0.8128 | 344.1300 | 0.3533 | 0.7761 |
| clip weights at 3 | WCS.homo | 300 | 0.0821 | 0.0212 | 0.1967 | 0.8130 | 344.2033 | 0.3533 | 0.7761 |
| clip weights at 3 | WCS.dtm | 300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3533 | 0.7761 |

## Interpretation

The important comparison is between `oracle weights` and `ignore shift`. If the oracle-weighted method has
mean FDP near or below q while the unweighted version has mean FDP above q, this shows that the paper's
guarantee is genuinely about weighted covariate shift adjustment, not merely about applying conformal
p-values mechanically.

This is a useful criticism/understanding point:

> The method is robust to arbitrary outcome prediction models, but it still relies on good covariate-shift
> weights. In practice, poor density-ratio estimation or aggressive clipping can compromise FDR control.

## Outputs

- `raw_results.csv`: all repetitions.
- `summary.csv`: aggregated results.
- `mean_fdp_by_weighting.png`: mean FDP by method and weighting strategy.
- `fdp_exceedance_by_weighting.png`: probability that realized FDP exceeds q.
