# FDR vs realized FDP demo

Configuration: outlier simulation results with `sig_id=4` and `out_prop_id=2`.
Nominal FDR level: `q=0.1`.

Interpretation:

- FDR is `E[FDP]`, the expectation over repeated experiments.
- It does not require every realized experiment to satisfy `FDP <= q`.
- A method can have mean FDP below q while a nontrivial fraction of individual runs exceed q.

Summary:

| method | runs | mean_fdp | sd_fdp | median_fdp | p90_fdp | max_fdp | prob_fdp_gt_q | mean_power | mean_nsel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WBH | 100 | 0.0819 | 0.0252 | 0.0820 | 0.1135 | 0.1441 | 0.2400 | 0.9753 | 212.6300 |
| WCS.hete | 100 | 0.0821 | 0.0252 | 0.0820 | 0.1131 | 0.1429 | 0.2600 | 0.9753 | 212.6900 |
| WCS.homo | 100 | 0.0821 | 0.0252 | 0.0820 | 0.1131 | 0.1429 | 0.2600 | 0.9753 | 212.6900 |
| WCS.dete | 100 | 0.0819 | 0.0251 | 0.0820 | 0.1131 | 0.1429 | 0.2400 | 0.9753 | 212.6300 |

See `fdp_histograms.png` for the distribution of realized FDP across seeds.
