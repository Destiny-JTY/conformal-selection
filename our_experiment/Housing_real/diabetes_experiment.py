#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验二: 真实数据实验 —— 协变量驱动的分布偏移

两种偏移模式:
  type=1: 正相关特征越大 → 越容易进校准集
  type=2: 负相关特征越大 → 越容易进校准集
  (自动用线性回归系数确定正/负相关特征)

用法:
  python diabetes_experiment.py <seed> <q> <type>
  例: python diabetes_experiment.py 42 2 1
"""

import numpy as np
import pandas as pd
import sys
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing

# ============================================================================
# 0. 方法实现
# ============================================================================


def weighted_BH(cal_scores, cal_weights, test_scores, test_weights, q=0.1):
    n_cal, n_test = len(cal_scores), len(test_scores)
    if n_test == 0:
        return np.array([], dtype=int)
    sum_w = np.sum(cal_weights)
    all_s = np.concatenate([cal_scores, test_scores])
    all_w = np.concatenate([cal_weights, test_weights])
    is_cal = np.concatenate([np.ones(n_cal), np.zeros(n_test)])
    order = np.argsort(all_s)
    all_s, all_w, is_cal = all_s[order], all_w[order], is_cal[order]
    test_pos = np.where(is_cal == 0)[0]
    pvals = np.zeros(n_test)
    for idx, pos in enumerate(test_pos):
        w_below = np.sum(all_w[:pos] * is_cal[:pos])
        w_eq = np.sum(all_w[pos] * is_cal[pos]) + all_w[pos]
        pvals[idx] = (w_below + w_eq * np.random.uniform()) / (sum_w + all_w[pos])
    order_p = np.argsort(pvals)
    thresh = q * np.arange(1, n_test + 1) / n_test
    below = pvals[order_p] <= thresh
    if not np.any(below):
        return np.array([], dtype=int)
    return order_p[:np.max(np.where(below)[0]) + 1]


def weighted_CS(cal_scores, cal_weights, test_scores, test_weights, q=0.1):
    n_cal, n_test = len(cal_scores), len(test_scores)
    sum_w = np.sum(cal_weights)
    if n_test == 0:
        return (np.array([], dtype=int),) * 4
    Rj_sizes = np.zeros(n_test)
    w_pvals = np.zeros(n_test)
    xis = np.random.uniform(size=n_test)
    for j in range(n_test):
        pval_j = np.zeros(n_test)
        for k in range(n_test):
            if k != j:
                pval_j[k] = (np.sum(cal_weights[cal_scores < test_scores[k]])
                             + test_weights[k] * (test_scores[j] < test_scores[k]))
            else:
                pval_j[k] = np.sum(cal_weights[cal_scores < test_scores[k]])
        pval_j = pval_j / (sum_w + test_weights[j])
        w_pvals[j] = (np.sum(cal_weights[cal_scores < test_scores[j]])
                      + (np.sum(cal_weights[cal_scores == test_scores[j]])
                         + test_weights[j]) * np.random.uniform()
                      ) / (sum_w + test_weights[j])
        order_j = np.argsort(pval_j)
        thresh = q * np.arange(1, n_test + 1) / n_test
        below_j = pval_j[order_j] <= thresh
        if np.any(below_j):
            Rj_sizes[j] = np.max(np.where(below_j)[0]) + 1
    Cj = q * Rj_sizes / n_test
    sel0 = np.where(w_pvals <= Cj)[0]
    if len(sel0) == 0:
        return (np.array([], dtype=int),) * 4
    df = pd.DataFrame({'id': range(n_test), 'pval': w_pvals, 'Cj': Cj,
                       'hete_Rj': Rj_sizes * xis,
                       'homo_Rj': Rj_sizes * np.random.uniform(),
                       'Rj': Rj_sizes})
    sub = df[df['pval'] <= df['Cj']]

    def prune(col):
        s = sub.sort_values(col)
        s['thresh'] = np.arange(1, len(s) + 1)
        below = s[col] <= s['thresh']
        if not np.any(below):
            return np.array([], dtype=int)
        return s['id'].values[:np.max(np.where(below)[0]) + 1]

    return sel0, prune('hete_Rj').astype(int), \
        prune('homo_Rj').astype(int), prune('Rj').astype(int)


def unweighted_BH(cal_scores, test_scores, q=0.1):
    n_cal, n_test = len(cal_scores), len(test_scores)
    if n_test == 0:
        return np.array([], dtype=int)
    pvals = np.array([(np.sum(cal_scores < test_scores[i])
                       + np.random.uniform() * (1 + np.sum(cal_scores == test_scores[i]))
                       ) / (n_cal + 1) for i in range(n_test)])
    order = np.argsort(pvals)
    thresh = q * np.arange(1, n_test + 1) / n_test
    below = pvals[order] <= thresh
    if not np.any(below):
        return np.array([], dtype=int)
    return order[:np.max(np.where(below)[0]) + 1]


def unweighted_CS(cal_scores, test_scores, q=0.1):
    n_cal, n_test = len(cal_scores), len(test_scores)
    _, hete, homo, dtm = weighted_CS(cal_scores, np.ones(n_cal),
                                     test_scores, np.ones(n_test), q)
    return hete, homo, dtm


def eval_selection(sel_idx, y_true, c_true):
    n_sel = len(sel_idx)
    total_pos = np.sum(y_true > c_true)
    if n_sel == 0:
        return 0.0, (0.0 if total_pos > 0 else 0.0), 0
    fp = np.sum(y_true[sel_idx] <= c_true[sel_idx])
    tp = np.sum(y_true[sel_idx] > c_true[sel_idx])
    return fp / n_sel, (tp / total_pos if total_pos > 0 else 0.0), n_sel


# ============================================================================
# 1. 加载数据, 线性模型获取系数
# ============================================================================
print(">>> 加载 California Housing, 拟合线性模型获取特征系数")
data = fetch_california_housing()
X_all = data['data'].astype(float)
y_all = data['target'].astype(float)
feat_names = data['feature_names']

# 使用全部数据 (不截断)

X_all = (X_all - X_all.mean(axis=0)) / (X_all.std(axis=0) + 1e-8)
n_total = X_all.shape[0]

lm_full = LinearRegression().fit(X_all, y_all)
coefs = lm_full.coef_
for i, (name, c) in enumerate(zip(feat_names, coefs)):
    print(f"      [{i}] {name:<12}: {c:+.4f}")

# ============================================================================
# 2. 参数 & 偏移类型
# ============================================================================
seed       = int(sys.argv[1])
q          = int(sys.argv[2]) / 10
shift_type = int(sys.argv[3])

np.random.seed(seed)

if shift_type == 1:
    # 取系数最大的 4 个正相关特征
    top_idx = np.argsort(coefs)[-4:][::-1]
else:
    # 取系数最小(最负)的 4 个负相关特征
    top_idx = np.argsort(coefs)[:4]
top_names = [feat_names[i] for i in top_idx]
top_coefs = coefs[top_idx]
tag = "正" if shift_type == 1 else "负"
print(f"\n>>> 偏移模式: {tag}相关特征组合 → 越大越容易进校准集")
for i, (n, c) in enumerate(zip(top_names, top_coefs)):
    print(f"      [{top_idx[i]}] {n:<12}: {c:+.4f}")

# ============================================================================
# 3. 制造协变量偏移 (多特征线性组合)
# ============================================================================
# 每个特征的权重正比于其系数绝对值
wgt = np.abs(top_coefs) / np.sum(np.abs(top_coefs))
logit_cal = -1.5 + 3.0 * (X_all[:, top_idx] @ wgt)
p_calib = np.clip(1.0 / (1.0 + np.exp(-logit_cal)), 0.1, 0.9)
in_calib = np.random.binomial(1, p_calib)

# 控制校准集大小
target_cal = 4000
cal_all = np.where(in_calib == 1)[0]
if len(cal_all) > target_cal:
    keep = np.random.choice(cal_all, target_cal, replace=False)
    in_calib[:] = 0
    in_calib[keep] = 1

target_test = 800     # 5:1 比例, 满足 q=0.2 时 min_p < BH 阈值
not_cal = np.where(in_calib == 0)[0]
test_idx = np.random.choice(not_cal, target_test, replace=False)

calib_X = X_all[in_calib == 1];  calib_y = y_all[in_calib == 1]
test_X  = X_all[test_idx];       test_y  = y_all[test_idx]

# 训练集: 从剩余中取 20% (同原论文 DTI)
remain_idx = np.array([i for i in range(n_total)
                        if i not in set(test_idx) and in_calib[i] == 0])
target_train = int(n_total * 0.20)
train_sel = np.random.choice(remain_idx, target_train, replace=False)
train_X = X_all[train_sel];  train_y = y_all[train_sel]

n_calib, n_test, n_train = len(calib_y), len(test_y), len(train_y)
shift_score_cal = calib_X[:, top_idx] @ wgt
shift_score_tst = test_X[:, top_idx] @ wgt
print(f"    校准: {n_calib} (偏移得分均值={shift_score_cal.mean():.2f}), "
      f"训练: {n_train}, 测试: {n_test} (偏移得分均值={shift_score_tst.mean():.2f})")

# ============================================================================
# 4. 选择阈值 & 权重 (用真实 P(calib|X), 同论文 simu.R 用真实 e(x))
# ============================================================================
c_thresh = np.quantile(train_y, 0.70)
n_pos = np.sum(test_y > c_thresh)
print(f"    阈值 c=70%分位数={c_thresh:.3f}, 测试集 Y>c: {n_pos}/{n_test}")

# 直接用 step 3 的真实 P(calib|X) 算权重, 不需要估计
e_calib_true = p_calib[in_calib == 1]
e_test_true  = p_calib[test_idx]
w_calib = 1.0 / e_calib_true
w_test  = 1.0 / e_test_true
w_calib_unif = np.ones(n_calib)
w_test_unif = np.ones(n_test)
print(f"    真实权重 w=1/P(calib|X): 校准 mean={w_calib.mean():.2f} "
      f"(range [{w_calib.min():.2f},{w_calib.max():.2f}]), "
      f"测试 mean={w_test.mean():.2f} "
      f"(range [{w_test.min():.2f},{w_test.max():.2f}])")

# ============================================================================
# 5. 训练预测模型 (RF) + 得分
# ============================================================================
print(f">>> 训练随机森林 (n_train={n_train})")
model = RandomForestRegressor(n_estimators=100, max_depth=8,
                               min_samples_leaf=5, random_state=seed)
model.fit(train_X, train_y)
mu_calib = model.predict(calib_X)
mu_test  = model.predict(test_X)

cal_score  = calib_y - mu_calib
test_score = c_thresh - mu_test

# ============================================================================
# 6. 运行 & 输出
# ============================================================================
print(f">>> 运行 8 方法, FDR={q}\n")
results = []
c_arr = np.full(n_test, c_thresh)

sel = weighted_BH(cal_score, w_calib, test_score, w_test, q)
results.append(["regression", "wBH", "Yes"] + list(eval_selection(sel, test_y, c_arr)))

_, hete, homo, dtm = weighted_CS(cal_score, w_calib, test_score, w_test, q)
for sname, s in [("wCS.hete", hete), ("wCS.homo", homo), ("wCS.dtm", dtm)]:
    results.append(["regression", sname, "Yes"] + list(eval_selection(s, test_y, c_arr)))

sel = unweighted_BH(cal_score, test_score, q)
results.append(["regression", "uBH", "No"] + list(eval_selection(sel, test_y, c_arr)))

uHe, uHo, uDt = unweighted_CS(cal_score, test_score, q)
for sname, s in [("uCS.hete", uHe), ("uCS.homo", uHo), ("uCS.dtm", uDt)]:
    results.append(["regression", sname, "No"] + list(eval_selection(s, test_y, c_arr)))

df = pd.DataFrame(results, columns=["Score", "Method", "Weighted", "FDP", "Power", "Nsel"])
df["seed"] = seed; df["q"] = q; df["shift_type"] = shift_type
df["shift_feature"] = "+".join(top_names); df["c_thresh"] = c_thresh

print("=" * 70)
print(f"  | FDR={q} | c={c_thresh:.3f} | seed={seed} | "
      f"{tag}相关偏移({' + '.join(top_names)})")
print("=" * 70)
print(f"  校准 {n_calib} | 训练 {n_train} | 测试 {n_test}")
print(f"  测试集 Y>c: {n_pos}")
print("=" * 70)
print()

sub = df[df["Score"] == "regression"]
print(f"{'Method':<12} {'Weighted':<8} {'FDP':>8} {'Power':>8} {'Nsel':>8}")
print("-" * 48)
for _, row in sub.iterrows():
    print(f"{row['Method']:<12} {row['Weighted']:<8} "
          f"{row['FDP']:>8.4f} {row['Power']:>8.4f} {int(row['Nsel']):>8d}")
print()

print("── 加权 vs 不加权 FDP 差异 ──")
print(f"{'Method':<12} {'FDP_w':>8} {'FDP_u':>8} {'Delta':>8}")
print("-" * 40)
for bm in ["BH", "CS.hete", "CS.homo", "CS.dtm"]:
    fw = df.loc[df["Method"] == f"w{bm}", "FDP"].values
    fu = df.loc[df["Method"] == f"u{bm}", "FDP"].values
    if len(fw) > 0 and len(fu) > 0:
        print(f"{bm:<12} {fw[0]:>8.4f} {fu[0]:>8.4f} {fu[0]-fw[0]:>+8.4f}")

save_dir = "./results_qsar/"
os.makedirs(save_dir, exist_ok=True)
fname = f"{save_dir}/seed{seed}_q{int(q*10)}_type{shift_type}.csv"
df.to_csv(fname, index=False)
print(f"\n已保存: {fname}")
