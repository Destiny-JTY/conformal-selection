#!/usr/bin/env python3
"""Experiment 2 visualization"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.size'] = 11

t1 = pd.read_csv("./results_qsar/summary_type1.csv")
t2 = pd.read_csv("./results_qsar/summary_type2.csv")
t1['shift'] = 'Type 1: Positive shift'
t2['shift'] = 'Type 2: Negative shift'
df = pd.concat([t1, t2], ignore_index=True)
df['Method_short'] = df['Method'].str.replace('w', 'W').str.replace('u', 'U')

GROUPS  = ['UBH', 'WBH', 'WCS.hete', 'WCS.homo', 'WCS.dtm']
LABELS  = ['BH', 'WBH', 'WCS.hete', 'WCS.homo', 'WCS.dtm']
width   = 0.18

# ============================================================================
# Fig 1: FDP
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
for ax, (sl, sub) in zip(axes, df.groupby('shift')):
    x = range(len(GROUPS))
    for i, qv in enumerate([0.1, 0.2]):
        offset = (i - 0.5) * width
        vals = []
        for g in GROUPS:
            rows = sub[(sub['Method_short'] == g) & (sub['q'] == qv)]
            vals.append(rows['fdp_mean'].values[0] if len(rows) > 0 else 0)
        bars = ax.bar([xi + offset for xi in x], vals, width,
                      label='q=%.1f' % qv, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0.005:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.02,
                        '%.3f' % h, ha='center', fontsize=7, fontweight='bold')
    ax.axhline(y=0.1, color='grey', ls='--', lw=0.8, alpha=0.5)
    ax.axhline(y=0.2, color='grey', ls='--', lw=0.8, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_title(sl, fontsize=11); ax.set_ylabel('FDP'); ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
fig.suptitle('FDP: wBH fails (>>q), all others control', fontweight='bold')
plt.tight_layout()
plt.savefig('./results_qsar/fig1_fdp.png', dpi=150, bbox_inches='tight')
plt.close(); print("Fig1 saved")

# ============================================================================
# Fig 2: Power
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
for ax, (sl, sub) in zip(axes, df.groupby('shift')):
    x = range(len(GROUPS))
    for i, qv in enumerate([0.1, 0.2]):
        offset = (i - 0.5) * width
        vals = []
        for g in GROUPS:
            rows = sub[(sub['Method_short'] == g) & (sub['q'] == qv)]
            vals.append(rows['power_mean'].values[0] if len(rows) > 0 else 0)
        bars = ax.bar([xi + offset for xi in x], vals, width,
                      label='q=%.1f' % qv, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0.005:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                        '%.3f' % h, ha='center', fontsize=7, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_title(sl, fontsize=11); ax.set_ylabel('Power'); ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
fig.suptitle('Power: CS > BH, weighted >= unweighted', fontweight='bold')
plt.tight_layout()
plt.savefig('./results_qsar/fig2_power.png', dpi=150, bbox_inches='tight')
plt.close(); print("Fig2 saved")

# ============================================================================
# Fig 3: Tradeoff
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
colors  = {'WBH': '#d62728', 'UBH': '#ff7f0e', 'WCS': '#1f77b4', 'UCS': '#2ca02c'}
markers = {'WBH': 'X', 'UBH': 's', 'WCS': 'o', 'UCS': 'D'}

for _, row in df.iterrows():
    m = row['Method']
    if 'wBH' in m: grp = 'WBH'
    elif 'uBH' in m: grp = 'UBH'
    elif 'wCS' in m: grp = 'WCS'
    elif 'uCS' in m: grp = 'UCS'
    else: continue
    if grp in ['UBH', 'UCS']:
        continue  # skip unweighted CS in tradeoff (too cluttered)
    ax.scatter(row['fdp_mean'], row['power_mean'], c=colors[grp],
              marker=markers[grp], s=120, alpha=0.85, edgecolors='black', lw=0.5)
    ax.annotate(grp, (row['fdp_mean'], row['power_mean']),
               fontsize=7, alpha=0.7,
               xytext=(row['fdp_mean']+0.01, row['power_mean']+0.008))

ax.axvline(x=0.1, color='grey', ls='--', lw=0.8, alpha=0.5)
ax.axvline(x=0.2, color='grey', ls='--', lw=0.8, alpha=0.5)
ax.set_xlabel('FDP'); ax.set_ylabel('Power')
ax.set_title('FDP-Power Trade-off', fontweight='bold')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('./results_qsar/fig3_tradeoff.png', dpi=150, bbox_inches='tight')
plt.close(); print("Fig3 saved")

# ============================================================================
# Fig 4: Nsel
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
for ax, (sl, sub) in zip(axes, df.groupby('shift')):
    x = range(len(GROUPS))
    for i, qv in enumerate([0.1, 0.2]):
        offset = (i - 0.5) * width
        vals = []
        for g in GROUPS:
            rows = sub[(sub['Method_short'] == g) & (sub['q'] == qv)]
            vals.append(rows['nsel_mean'].values[0] if len(rows) > 0 else 0)
        bars = ax.bar([xi + offset for xi in x], vals, width,
                      label='q=%.1f' % qv, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 1,
                    '%.0f' % h, ha='center', fontsize=7, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_title(sl, fontsize=11); ax.set_ylabel('Nsel'); ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
fig.suptitle('Nsel: weighted methods', fontweight='bold')
plt.tight_layout()
plt.savefig('./results_qsar/fig4_nsel.png', dpi=150, bbox_inches='tight')
plt.close(); print("Fig4 saved")

# ============================================================================
# Summary table
# ============================================================================
print("\n" + "=" * 70)
print("  Experiment 2 Summary (20-seed mean)")
print("=" * 70)

for sl, sub in df.groupby('shift'):
    print("\n  %s" % sl)
    print("  " + "-" * 55)
    print("  %-12s %5s %8s %8s %8s %8s %8s" % (
        'Method', 'FDR', 'FDP_w', 'FDP_u', 'Power_w', 'Power_u', 'Nsel_w'))
    print("  " + "-" * 55)
    for qv in [0.1, 0.2]:
        for bm in ['BH', 'CS.hete', 'CS.homo', 'CS.dtm']:
            rw = sub[(sub['Method'] == 'w%s' % bm) & (sub['q'] == qv)]
            ru = sub[(sub['Method'] == 'u%s' % bm) & (sub['q'] == qv)]
            if len(rw) > 0 and len(ru) > 0:
                print("  %-12s %5.1f %8.4f %8.4f %8.4f %8.4f %8.1f" % (
                    bm, qv,
                    rw['fdp_mean'].values[0], ru['fdp_mean'].values[0],
                    rw['power_mean'].values[0], ru['power_mean'].values[0],
                    rw['nsel_mean'].values[0]))

print("\nFigures: ./results_qsar/")
