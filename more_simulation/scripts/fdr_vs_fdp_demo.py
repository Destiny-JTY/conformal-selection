#!/usr/bin/env python
"""Demonstrate that FDR controls E[FDP], not every realized FDP."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = (
    ROOT
    / "reference"
    / "conformal-selection"
    / "experiments"
    / "outlier_detection"
    / "simulations"
    / "results"
)
OUT_DIR = ROOT / "outputs" / "fdr_vs_fdp_demo"
NAME_RE = re.compile(r"seed(?P<seed>\d+)sig(?P<sig_id>\d+)prop(?P<prop>\d+)\.csv$")


def load_results(result_dir: Path) -> pd.DataFrame:
    rows = []
    for path in result_dir.glob("seed*sig*prop*.csv"):
        match = NAME_RE.match(path.name)
        if not match:
            continue
        data = pd.read_csv(path, index_col=0)
        data["seed_from_file"] = int(match.group("seed"))
        data["sig_id"] = int(match.group("sig_id"))
        data["out_prop_id"] = int(match.group("prop"))
        rows.append(data)
    if not rows:
        raise FileNotFoundError(f"No result CSV files found in {result_dir}")
    return pd.concat(rows, ignore_index=True)


def summarize(df: pd.DataFrame, q: float) -> pd.DataFrame:
    grouped = df.groupby("method", sort=False)
    return grouped.agg(
        runs=("fdp", "size"),
        mean_fdp=("fdp", "mean"),
        sd_fdp=("fdp", "std"),
        median_fdp=("fdp", "median"),
        p90_fdp=("fdp", lambda x: x.quantile(0.90)),
        max_fdp=("fdp", "max"),
        prob_fdp_gt_q=("fdp", lambda x: (x > q).mean()),
        mean_power=("power", "mean"),
        mean_nsel=("nsel", "mean"),
    ).reset_index()


def plot_histograms(df: pd.DataFrame, summary: pd.DataFrame, q: float, out_path: Path) -> None:
    methods = list(summary["method"])
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    axes = axes.ravel()
    bins = [i / 100 for i in range(0, 41, 2)]
    for ax, method in zip(axes, methods):
        sub = df[df["method"] == method]
        mean_fdp = float(sub["fdp"].mean())
        exceed = float((sub["fdp"] > q).mean())
        ax.hist(sub["fdp"], bins=bins, color="#4C78A8", alpha=0.85, edgecolor="white")
        ax.axvline(q, color="#D62728", linestyle="--", linewidth=1.5, label=f"q={q:.2f}")
        ax.axvline(mean_fdp, color="#2CA02C", linestyle="-", linewidth=1.5, label="mean FDP")
        ax.set_title(f"{method}: mean={mean_fdp:.3f}, P(FDP>q)={exceed:.2f}")
        ax.set_xlabel("Realized FDP")
        ax.set_ylabel("Runs")
        ax.legend(fontsize=8)
    fig.suptitle("FDR is the expectation of FDP; single-run FDP can exceed q", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(df: pd.DataFrame, summary: pd.DataFrame, q: float, out_path: Path) -> None:
    sig_id = int(df["sig_id"].iloc[0])
    out_prop_id = int(df["out_prop_id"].iloc[0])
    table_columns = list(summary.columns)
    table = [
        "| " + " | ".join(table_columns) + " |",
        "| " + " | ".join(["---"] * len(table_columns)) + " |",
    ]
    for _, row in summary.iterrows():
        values = []
        for col in table_columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        table.append("| " + " | ".join(values) + " |")

    lines = [
        "# FDR vs realized FDP demo",
        "",
        f"Configuration: outlier simulation results with `sig_id={sig_id}` and `out_prop_id={out_prop_id}`.",
        f"Nominal FDR level: `q={q}`.",
        "",
        "Interpretation:",
        "",
        "- FDR is `E[FDP]`, the expectation over repeated experiments.",
        "- It does not require every realized experiment to satisfy `FDP <= q`.",
        "- A method can have mean FDP below q while a nontrivial fraction of individual runs exceed q.",
        "",
        "Summary:",
        "",
        "\n".join(table),
        "",
        "See `fdp_histograms.png` for the distribution of realized FDP across seeds.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sig-id", type=int, default=4, help="File-level signal id, e.g. 4 for seed*sig4prop2.csv")
    parser.add_argument("--out-prop-id", type=int, default=2, help="File-level outlier proportion id")
    parser.add_argument("--q", type=float, default=0.1)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ns = parser.parse_args()

    df = load_results(ns.result_dir)
    df = df[(df["sig_id"] == ns.sig_id) & (df["out_prop_id"] == ns.out_prop_id)].copy()
    if df.empty:
        raise ValueError(f"No rows for sig_id={ns.sig_id}, out_prop_id={ns.out_prop_id}")

    ns.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(ns.out_dir / "selected_runs.csv", index=False)
    summary = summarize(df, ns.q)
    summary.to_csv(ns.out_dir / "summary.csv", index=False)
    plot_histograms(df, summary, ns.q, ns.out_dir / "fdp_histograms.png")
    write_report(df, summary, ns.q, ns.out_dir / "README.md")
    print(summary.to_string(index=False))
    print(f"Wrote outputs to {ns.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
