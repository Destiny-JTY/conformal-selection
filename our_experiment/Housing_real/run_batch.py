#!/usr/bin/env python3
"""实验二批处理: 20 seed × 2 FDR × 2 偏移类型"""
import subprocess, glob, os
import pandas as pd

seeds = list(range(1, 21))
q_levels = [1, 2]       # 1=0.1, 2=0.2
shift_types = [1, 2]    # 1=正相关偏移, 2=负相关偏移

for st in shift_types:
    for qv in q_levels:
        for seed in seeds:
            cmd = f"python diabetes_experiment.py {seed} {qv} {st}"
            print(f"\n>>> [type={st}, q=0.{qv}, seed={seed}] {cmd}")
            subprocess.run(cmd, shell=True)

# 汇总
print("\n>>> 汇总...")
files = glob.glob("./results_qsar/seed*_q*_type*.csv")
if not files:
    print("无结果文件")
    exit(1)

all_data = pd.concat([pd.read_csv(f) for f in files])

for st in [1, 2]:
    st_name = "正相关偏移" if st == 1 else "负相关偏移"
    data_st = all_data[all_data["shift_type"] == st]
    print(f"\n{'='*70}")
    print(f"  {st_name} (shift_feature={data_st['shift_feature'].iloc[0]})")
    print(f"{'='*70}")

    summary = data_st.groupby(["Score", "Method", "Weighted", "q"]).agg(
        n_runs=("FDP", "count"),
        fdp_mean=("FDP", "mean"), fdp_sd=("FDP", "std"),
        power_mean=("Power", "mean"), power_sd=("Power", "std"),
        nsel_mean=("Nsel", "mean"), nsel_sd=("Nsel", "std"),
    ).reset_index()

    summary.to_csv(f"./results_qsar/summary_type{st}.csv", index=False)

    print(f"{'Score':<14} {'q':>5} {'Method':<12} {'FDP_w':>8} {'FDP_u':>8} {'Delta':>8}")
    print("-" * 60)
    for qq in sorted(summary["q"].unique()):
        sub = summary[(summary["q"] == qq)]
        for bm in ["BH", "CS.hete", "CS.homo", "CS.dtm"]:
            rw = sub[(sub["Method"] == f"w{bm}")]
            ru = sub[(sub["Method"] == f"u{bm}")]
            if len(rw) > 0 and len(ru) > 0:
                print(f"{'regression':<14} {qq:>5.1f} {bm:<12} "
                      f"{rw['fdp_mean'].values[0]:>8.4f} "
                      f"{ru['fdp_mean'].values[0]:>8.4f} "
                      f"{ru['fdp_mean'].values[0]-rw['fdp_mean'].values[0]:>+8.4f}")

print("\n完成.")
