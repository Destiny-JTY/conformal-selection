#!/usr/bin/env python
"""Simulation showing WCS depends on correct covariate-shift weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "weight_misspecification_demo"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def bh_from_pvals(pvals: np.ndarray, q: float) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals, kind="mergesort")
    thresholds = q * np.arange(1, n + 1) / n
    passed = np.flatnonzero(pvals[order] <= thresholds)
    if len(passed) == 0:
        return np.array([], dtype=int)
    return order[: passed[-1] + 1]


def weighted_bh(
    cal_scores: np.ndarray,
    cal_weights: np.ndarray,
    test_scores: np.ndarray,
    test_weights: np.ndarray,
    q: float,
    rng: np.random.Generator,
) -> np.ndarray:
    cal_scores = np.asarray(cal_scores).ravel()
    cal_weights = np.asarray(cal_weights).ravel()
    test_scores = np.asarray(test_scores).ravel()
    test_weights = np.asarray(test_weights).ravel()
    order = np.argsort(cal_scores, kind="mergesort")
    sorted_scores = cal_scores[order]
    sorted_weights = cal_weights[order]
    cum_weights = np.concatenate(([0.0], np.cumsum(sorted_weights)))
    lt_pos = np.searchsorted(sorted_scores, test_scores, side="left")
    le_pos = np.searchsorted(sorted_scores, test_scores, side="right")
    cal_lt = cum_weights[lt_pos]
    cal_eq = cum_weights[le_pos] - cum_weights[lt_pos]
    pvals = (cal_lt + (cal_eq + test_weights) * rng.uniform(size=len(test_scores))) / (
        np.sum(cal_weights) + test_weights
    )
    return bh_from_pvals(pvals, q)


def weighted_cs(
    cal_scores: np.ndarray,
    cal_weights: np.ndarray,
    test_scores: np.ndarray,
    test_weights: np.ndarray,
    q: float,
    rng: np.random.Generator,
    rand: str = "hete",
) -> np.ndarray:
    cal_scores = np.asarray(cal_scores).ravel()
    cal_weights = np.asarray(cal_weights).ravel()
    test_scores = np.asarray(test_scores).ravel()
    test_weights = np.asarray(test_weights).ravel()
    ntest = len(test_scores)
    sum_cal = float(np.sum(cal_weights))

    order = np.argsort(cal_scores, kind="mergesort")
    sorted_scores = cal_scores[order]
    sorted_weights = cal_weights[order]
    cum_weights = np.concatenate(([0.0], np.cumsum(sorted_weights)))
    lt_pos = np.searchsorted(sorted_scores, test_scores, side="left")
    le_pos = np.searchsorted(sorted_scores, test_scores, side="right")
    cal_lt = cum_weights[lt_pos]
    cal_eq = cum_weights[le_pos] - cum_weights[lt_pos]

    rj_sizes = np.zeros(ntest)
    w_pvals = np.zeros(ntest)
    thresholds = q * np.arange(1, ntest + 1) / ntest

    for j in range(ntest):
        pval_j = cal_lt + test_weights * (test_scores[j] < test_scores)
        pval_j[j] = 0.0
        pval_j = pval_j / (sum_cal + test_weights[j])
        sorted_pvals = np.sort(pval_j)
        passed = np.flatnonzero(sorted_pvals <= thresholds)
        rj_sizes[j] = passed[-1] + 1 if len(passed) else 0
        w_pvals[j] = (cal_lt[j] + (cal_eq[j] + test_weights[j]) * rng.uniform()) / (
            sum_cal + test_weights[j]
        )

    selected_first = w_pvals <= q * rj_sizes / ntest
    if not np.any(selected_first):
        return np.array([], dtype=int)

    if rand == "hete":
        xi = rng.uniform(size=ntest)
    elif rand == "homo":
        xi = np.repeat(rng.uniform(), ntest)
    elif rand == "dtm":
        xi = np.ones(ntest)
    else:
        raise ValueError(f"Unknown pruning mode: {rand}")

    xi_r = xi * rj_sizes
    xi_r[~selected_first] = ntest + 1
    order = np.argsort(xi_r, kind="mergesort")
    passed = np.flatnonzero(xi_r[order] < np.arange(1, ntest + 1))
    if len(passed) == 0:
        return np.array([], dtype=int)
    return order[: passed[-1] + 1]


def evaluate(sel: np.ndarray, y_test: np.ndarray) -> tuple[float, float, int]:
    if len(sel) == 0:
        return 0.0, 0.0, 0
    false_discoveries = np.sum(y_test[sel] == 0)
    positives = np.sum(y_test == 1)
    true_discoveries = np.sum(y_test[sel] == 1)
    return false_discoveries / len(sel), true_discoveries / max(positives, 1), len(sel)


def run_one(
    seed: int,
    n_cal: int,
    n_test: int,
    delta: float,
    beta0: float,
    beta1: float,
    q: float,
    clip: float,
) -> list[dict[str, float | int | str]]:
    rng = np.random.default_rng(seed)
    x_cal = rng.normal(0.0, 1.0, size=n_cal)
    x_test = rng.normal(delta, 1.0, size=n_test)
    mu_cal = sigmoid(beta0 + beta1 * x_cal)
    mu_test = sigmoid(beta0 + beta1 * x_test)
    y_cal = rng.binomial(1, mu_cal)
    y_test = rng.binomial(1, mu_test)

    true_w_cal = np.exp(delta * x_cal - 0.5 * delta**2)
    true_w_test = np.exp(delta * x_test - 0.5 * delta**2)
    wrong_w_cal = np.ones_like(true_w_cal)
    wrong_w_test = np.ones_like(true_w_test)
    clipped_w_cal = np.minimum(true_w_cal, clip)
    clipped_w_test = np.minimum(true_w_test, clip)

    # Clipped classification score from the paper. Positives get a large score;
    # nulls use -mu(x). Test threshold is c=0, so test score is -mu(x).
    m_const = 100.0
    cal_scores = m_const * y_cal - mu_cal
    test_scores = -mu_test

    weight_sets = {
        "oracle weights": (true_w_cal, true_w_test),
        "ignore shift": (wrong_w_cal, wrong_w_test),
        f"clip weights at {clip:g}": (clipped_w_cal, clipped_w_test),
    }
    rows: list[dict[str, float | int | str]] = []
    for weight_label, (w_cal, w_test) in weight_sets.items():
        for method in ["WBH", "WCS.hete", "WCS.homo", "WCS.dtm"]:
            if method == "WBH":
                sel = weighted_bh(cal_scores, w_cal, test_scores, w_test, q, rng)
            else:
                rand = method.split(".")[1]
                sel = weighted_cs(cal_scores, w_cal, test_scores, w_test, q, rng, rand=rand)
            fdp, power, nsel = evaluate(sel, y_test)
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "weights": weight_label,
                    "fdp": fdp,
                    "power": power,
                    "nsel": nsel,
                    "test_positive_rate": float(np.mean(y_test)),
                    "cal_positive_rate": float(np.mean(y_cal)),
                    "mean_cal_weight": float(np.mean(w_cal)),
                    "max_cal_weight": float(np.max(w_cal)),
                }
            )
    return rows


def summarize(df: pd.DataFrame, q: float) -> pd.DataFrame:
    return (
        df.groupby(["weights", "method"], sort=False)
        .agg(
            runs=("fdp", "size"),
            mean_fdp=("fdp", "mean"),
            sd_fdp=("fdp", "std"),
            p90_fdp=("fdp", lambda x: x.quantile(0.90)),
            max_fdp=("fdp", "max"),
            prob_fdp_gt_q=("fdp", lambda x: (x > q).mean()),
            mean_power=("power", "mean"),
            mean_nsel=("nsel", "mean"),
            mean_cal_positive_rate=("cal_positive_rate", "mean"),
            mean_test_positive_rate=("test_positive_rate", "mean"),
        )
        .reset_index()
    )


def plot_summary(summary: pd.DataFrame, q: float, out_dir: Path) -> None:
    methods = ["WBH", "WCS.hete", "WCS.homo", "WCS.dtm"]
    weight_labels = list(summary["weights"].drop_duplicates())
    x = np.arange(len(methods))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, weight_label in enumerate(weight_labels):
        sub = summary[summary["weights"] == weight_label].set_index("method").loc[methods]
        ax.bar(x + (i - 1) * width, sub["mean_fdp"], width, label=weight_label)
    ax.axhline(q, color="#D62728", linestyle="--", linewidth=1.5, label=f"q={q:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Mean FDP across repetitions")
    ax.set_title("Correct weights matter: FDR can fail under weight misspecification")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mean_fdp_by_weighting.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, weight_label in enumerate(weight_labels):
        sub = summary[summary["weights"] == weight_label].set_index("method").loc[methods]
        ax.bar(x + (i - 1) * width, sub["prob_fdp_gt_q"], width, label=weight_label)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("P(realized FDP > q)")
    ax.set_title("Single-run FDP exceedance becomes more common under wrong weights")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fdp_exceedance_by_weighting.png", dpi=180)
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    columns = [
        "weights",
        "method",
        "runs",
        "mean_fdp",
        "sd_fdp",
        "prob_fdp_gt_q",
        "mean_power",
        "mean_nsel",
        "mean_cal_positive_rate",
        "mean_test_positive_rate",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df[columns].iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> None:
    report = f"""# Weight misspecification simulation

## Motivation

This simulation targets a potential weakness of weighted conformal selection under covariate shift:
the finite-sample FDR guarantee depends on using the correct density-ratio weights. The procedure is
model-free with respect to the outcome model, but it is not weight-free. If the covariate-shift weights
are ignored or strongly distorted, the weighted conformal p-values may no longer be calibrated for the
test distribution.

## Data-generating mechanism

- Calibration covariate: `X_cal ~ N(0, 1)`.
- Test covariate: `X_test ~ N(delta, 1)` with `delta={args.delta}`.
- Outcome: `Y | X ~ Bernoulli(sigmoid(beta0 + beta1 X))`, with `beta0={args.beta0}` and `beta1={args.beta1}`.
- The test distribution is shifted toward larger X, where positives are more likely.
- We select test points believed to have `Y=1`; false discoveries are selected points with `Y=0`.

The true density ratio is:

```text
w(x) = dQ_X / dP_X = exp(delta * x - delta^2 / 2)
```

We compare three weighting strategies:

- oracle weights: use the true density ratio;
- ignore shift: set all weights to 1;
- clipped weights: use `min(w(x), {args.clip})`.

## Why this probes the paper's limitation

Under covariate shift, test null examples are not distributed like calibration null examples. If high-X nulls
are underrepresented in calibration and we ignore weights, the conformal p-values can become too small in the
shifted region. This can increase the false discovery proportion. Correct weights repair the comparison by
reweighting calibration examples toward the test covariate distribution.

## Results

Nominal FDR level: `q={args.q}`. Number of repetitions: `{args.reps}`.

{markdown_table(summary)}

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
"""
    (out_dir / "README.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--n-cal", type=int, default=800)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--delta", type=float, default=1.5)
    parser.add_argument("--beta0", type=float, default=-1.0)
    parser.add_argument("--beta1", type=float, default=2.0)
    parser.add_argument("--q", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=3.0)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for seed in range(1, args.reps + 1):
        all_rows.extend(
            run_one(
                seed=seed,
                n_cal=args.n_cal,
                n_test=args.n_test,
                delta=args.delta,
                beta0=args.beta0,
                beta1=args.beta1,
                q=args.q,
                clip=args.clip,
            )
        )
    raw = pd.DataFrame(all_rows)
    summary = summarize(raw, args.q)
    raw.to_csv(args.out_dir / "raw_results.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    plot_summary(summary, args.q, args.out_dir)
    write_report(summary, args, args.out_dir)
    print(summary.to_string(index=False))
    print(f"Wrote outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
