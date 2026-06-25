#!/usr/bin/env python3
"""实验二可视化: 正/负相关偏移下加权 vs 不加权对比"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.size'] = 11

# ============================================================================
# 加载
# ============================================================================
t1 = pd.read_csv("./results_qsar/summary_type1.csv")
t2 = pd.read_csv("./results_qsar/summary_type2.csv")
t1['shift'] = 'Type 1: MedInc (+0.77)'
t2['shift'] = 'Type 2: Latitude (-0.93)'
df = pd.concat([t1, t2], ignore_index=True)

# 整理
df['Method_short'] = df['Method'].str.replace('w', 'W').str.replace('u', 'U')
df['FDR_target'] = df['q'].apply(lambda x: f'q={x:.1f}')

# ============================================================================
# 图 1: FDP 对比 —— 加权 vs 不加权, 按偏移类型 × FDR 分面
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

for ax, (shift_label, sub) in zip(axes, df.groupby('shift')):
    methods = ['WBH', 'WCS.hete', 'WCS.homo', 'WCS.dtm',
               'UBH', 'UCS.hete', 'UCS.homo', 'UCS.dtm']
    colors = ['#d62728', '#1f77b4', '#1f77b4', '#1f77b4',
              '#ff7f0e', '#2ca02c', '#2ca02c', '#2ca02c']
    hatches = ['', '', '//', '\\\\', '', '', '//', '\\\\']

    x = range(len(methods))
    width = 0.35

    for i, q_val in enumerate([0.1, 0.2]):
        offset = (i - 0.5) * width
        vals = []
        for m in methods:
            row = sub[(sub['Method_short'] == m) & (sub['q'] == q_val)]
            vals.append(row['fdp_mean'].values[0] if len(row) > 0 else 0)
        bars = ax.bar([xi + offset for xi in x], vals, width,
                      label=f'q={q_val:.1f}', alpha=0.85)

    # 标注名义 FDR 线
    ax.axhline(y=0.1, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=0.2, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right')
    ax.set_title(shift_label)
    ax.set_ylabel('FDP')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('FDP: 加权(W) vs 不加权(U)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('./results_qsar/fig1_fdp_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("图1 已保存: fig1_fdp_comparison.png")

# ============================================================================
# 图 2: wBH vs 其他方法的 FDP (突出 wBH 失控)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (shift_label, sub) in zip(axes, df.groupby('shift')):
    summary_rows = []
    for q_val in [0.1, 0.2]:
        sub_q = sub[sub['q'] == q_val]
        for method_group, label in [('WBH', 'WBH'),
                                     ('WCS', 'WCS(avg)'),
                                     ('UBH', 'UBH'),
                                     ('UCS', 'UCS(avg)')]:
            if method_group == 'WBH':
                rows = sub_q[sub_q['Method_short'] == 'WBH']
            elif method_group == 'UBH':
                rows = sub_q[sub_q['Method_short'] == 'UBH']
            elif method_group == 'WCS':
                rows = sub_q[sub_q['Method_short'].isin(['WCS.hete', 'WCS.homo', 'WCS.dtm'])]
            elif method_group == 'UCS':
                rows = sub_q[sub_q['Method_short'].isin(['UCS.hete', 'UCS.homo', 'UCS.dtm'])]

            if len(rows) > 0:
                summary_rows.append({
                    'Method': label, 'q': q_val,
                    'fdp_mean': rows['fdp_mean'].mean(),
                    'fdp_sd': rows['fdp_sd'].mean()
                })

    sub_plot = pd.DataFrame(summary_rows)
    x = range(4)
    width = 0.35
    colors_bar = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c']

    for i, q_val in enumerate([0.1, 0.2]):
        offset = (i - 0.5) * width
        vals = [sub_plot[(sub_plot['Method'] == m) & (sub_plot['q'] == q_val)]['fdp_mean'].values[0]
                for m in ['WBH', 'WCS(avg)', 'UBH', 'UCS(avg)']]
        ax.bar([xi + offset for xi in x], vals, width,
               label=f'q={q_val:.1f}', alpha=0.85,
               color=['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c'] if i == 0 else None)

    ax.axhline(y=0.1, color='grey', linestyle='--', linewidth=0.8)
    ax.axhline(y=0.2, color='grey', linestyle='--', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(['WBH', 'WCS(avg)', 'UBH', 'UCS(avg)'])
    ax.set_title(shift_label)
    ax.set_ylabel('FDP')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('FDP 分组汇总: WBH 失控, WCS 受控', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('./results_qsar/fig2_fdp_grouped.png', dpi=150, bbox_inches='tight')
plt.close()
print("图2 已保存: fig2_fdp_grouped.png")

# ============================================================================
# 图 3: Power 对比
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, (shift_label, sub) in zip(axes, df.groupby('shift')):
    method_groups = ['WBH', 'WCS(avg)', 'UBH', 'UCS(avg)']
    x = range(4)
    width = 0.35

    for i, q_val in enumerate([0.1, 0.2]):
        offset = (i - 0.5) * width
        vals = []
        for mg in method_groups:
            if mg == 'WBH':
                rows = sub[(sub['Method_short'] == 'WBH') & (sub['q'] == q_val)]
            elif mg == 'UBH':
                rows = sub[(sub['Method_short'] == 'UBH') & (sub['q'] == q_val)]
            elif mg == 'WCS(avg)':
                rows = sub[(sub['Method_short'].isin(['WCS.hete', 'WCS.homo', 'WCS.dtm'])) & (sub['q'] == q_val)]
            elif mg == 'UCS(avg)':
                rows = sub[(sub['Method_short'].isin(['UCS.hete', 'UCS.homo', 'UCS.dtm'])) & (sub['q'] == q_val)]
            vals.append(rows['power_mean'].mean() if len(rows) > 0 else 0)
        ax.bar([xi + offset for xi in x], vals, width,
               label=f'q={q_val:.1f}', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(method_groups)
    ax.set_title(shift_label)
    ax.set_ylabel('Power')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('Power 对比: 加权 vs 不加权', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('./results_qsar/fig3_power.png', dpi=150, bbox_inches='tight')
plt.close()
print("图3 已保存: fig3_power.png")

# ============================================================================
# 图 4: delta_FDP = FDP_unweighted - FDP_weighted (仅 BH 方法对比)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 5))

delta_data = []
for shift_label, sub in df.groupby('shift'):
    for q_val in [0.1, 0.2]:
        w_row = sub[(sub['Method_short'] == 'WBH') & (sub['q'] == q_val)]
        u_row = sub[(sub['Method_short'] == 'UBH') & (sub['q'] == q_val)]
        if len(w_row) > 0 and len(u_row) > 0:
            delta_data.append({
                'shift': shift_label,
                'q': q_val,
                'FDP_w': w_row['fdp_mean'].values[0],
                'FDP_u': u_row['fdp_mean'].values[0],
                'delta': u_row['fdp_mean'].values[0] - w_row['fdp_mean'].values[0]
            })

delta_df = pd.DataFrame(delta_data)
x = range(len(delta_df))
colors = ['#d62728' if d < 0 else '#2ca02c' for d in delta_df['delta']]
bars = ax.bar(x, delta_df['delta'], color=colors, alpha=0.85, edgecolor='grey')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"{r['shift'].split(':')[0]}\nq={r['q']:.1f}"
                     for _, r in delta_df.iterrows()])
ax.set_ylabel('Delta FDP = UBH - WBH')
ax.set_title('UBH vs WBH 的 FDP 差异 (负值 = WBH 更差)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 标注数值
for i, (_, r) in enumerate(delta_df.iterrows()):
    label = f"{r['delta']:+.3f}"
    ax.text(i, r['delta'] + (0.02 if r['delta'] >= 0 else -0.05),
            label, ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('./results_qsar/fig4_delta_fdp.png', dpi=150, bbox_inches='tight')
plt.close()
print("图4 已保存: fig4_delta_fdp.png")

# ============================================================================
# 图 5: 汇总表格
# ============================================================================
print("\n========== 汇总表 ==========")
print(f"{'Shift':<20} {'q':>5} {'Method':>10} {'FDP_w':>8} {'FDP_u':>8} {'ΔFDP':>8} {'Power_w':>8} {'Power_u':>8}")
print("-" * 80)
for shift_label, sub in df.groupby('shift'):
    for q_val in [0.1, 0.2]:
        for base in ['BH']:
            w_row = sub[(sub['Method_short'] == 'WBH') & (sub['q'] == q_val)]
            u_row = sub[(sub['Method_short'] == 'UBH') & (sub['q'] == q_val)]
            if len(w_row) > 0 and len(u_row) > 0:
                print(f"{shift_label:<20} {q_val:>5.1f} {'BH':>10} "
                      f"{w_row['fdp_mean'].values[0]:>8.4f} "
                      f"{u_row['fdp_mean'].values[0]:>8.4f} "
                      f"{u_row['fdp_mean'].values[0]-w_row['fdp_mean'].values[0]:>+8.4f} "
                      f"{w_row['power_mean'].values[0]:>8.4f} "
                      f"{u_row['power_mean'].values[0]:>8.4f}")

print("\n所有图保存至 ./results_qsar/")
