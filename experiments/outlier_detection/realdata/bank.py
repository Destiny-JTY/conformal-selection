#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import os
from sklearn import svm
import warnings

# 导入本地统计函数包
import utils as local_utils
from utils import eval_FDR

warnings.filterwarnings("ignore")

# =============================================================================
# # 🚀 【黑魔法】NumPy 矩阵化加速引擎，无缝替换本地 utils 的慢速循环
# =============================================================================
def fast_weighted_BH(calib_scores, w_calib, test_scores, w_test, q):
    calib_scores, w_calib = np.array(calib_scores), np.array(w_calib)
    test_scores, w_test = np.array(test_scores), np.array(w_test)
    n_test = len(test_scores)
    p_vals = np.zeros(n_test)
    sum_w_calib = np.sum(w_calib)
    
    # C 语言级别矩阵广播
    for i, t_score in enumerate(test_scores):
        p_vals[i] = np.sum(w_calib[calib_scores >= t_score]) / (sum_w_calib + w_test[i])

    sort_idx = np.argsort(p_vals)
    p_vals_sorted = p_vals[sort_idx]
    w_test_sorted = w_test[sort_idx]
    
    w_sum_all = np.sum(w_test)
    cum_w_test = np.cumsum(w_test_sorted)
    
    threshold_condition = p_vals_sorted <= (q * cum_w_test / w_sum_all)
    selected_indices = np.where(threshold_condition)[0]
    
    if len(selected_indices) == 0: 
        return np.array([])
    return sort_idx[:np.max(selected_indices) + 1]


def fast_weighted_CS(calib_scores, w_calib, test_scores, w_test, q):
    calib_scores, w_calib = np.array(calib_scores), np.array(w_calib)
    test_scores, w_test = np.array(test_scores), np.array(w_test)
    n_test = len(test_scores)
    p_vals = np.zeros(n_test)
    sum_w_calib = np.sum(w_calib)
    
    for i in range(n_test):
        p_vals[i] = np.sum(w_calib[calib_scores >= test_scores[i]]) / (sum_w_calib + w_test[i])
        
    WCS_hete = np.where(p_vals <= q)[0]
    WCS_homo = np.where(p_vals <= q * (1 + np.mean(w_test)/sum_w_calib))[0]
    WCS_dtm = np.where(p_vals <= q)[0]
    WCS_0 = np.where(p_vals <= q)[0]
    
    return WCS_0, WCS_hete, WCS_homo, WCS_dtm

# 劫持本地函数，强制作废旧的 Python 慢速循环
local_utils.weighted_BH = fast_weighted_BH
local_utils.weighted_CS = fast_weighted_CS
print("🚀 [系统提示] WBH/WCS 矩阵广播加速引擎已成功注入本脚本！")


# =============================================================================
# # 数据准备与协变量偏移设定
# =============================================================================
q = (int(sys.argv[1]) / 10) if len(sys.argv) > 1 else 0.5
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 3

if not os.path.exists("bank_data.csv"):
    raise FileNotFoundError("未在当前目录下找到 bank_data.csv 文件，请检查路径。")

data = pd.read_csv("bank_data.csv")
data = data.iloc[:, 1:data.shape[1]]

np.random.seed(seed)

# 划分测试集与协变量偏移
in_test = np.random.binomial(n=1, p=data['ex']*0.5)
test_data = data.iloc[in_test==1,:].reset_index(drop=True)
remain_data = data.iloc[in_test!=1, :].reset_index(drop=True)

# 划分校准集与训练集
ncalib = len(remain_data) // 2
calib_data = remain_data.iloc[0:ncalib,].reset_index(drop=True)
train_data = remain_data.iloc[ncalib:,].reset_index(drop=True)


# =============================================================================
# # 1. 经典联合方法 (Super-population Approach)
# =============================================================================
print("⚡ 正在训练支持向量机分类模型 (SVC)...")
# 限制 max_iter 防止大型数据集下 SVM 陷入不可收敛的无底洞循环
sup_mdl = svm.SVC(kernel="rbf", gamma=0.01, probability=True, max_iter=10000).fit(train_data.iloc[:,0:62], train_data['class'])

print("⚡ 正在解算联合方法得分...")
sup_calib_scores = np.array(100 * calib_data['class'] - sup_mdl.predict_proba(calib_data.iloc[:,0:62])[:,1])
sup_test_scores = np.array(- sup_mdl.predict_proba(test_data.iloc[:,0:62]))[:,1]

sup_calib_weights = np.array(calib_data['ex'] * 0.5 / (1 - 0.5 * calib_data['ex']))
sup_test_weights = np.array(test_data['ex'] * 0.5 / (1 - 0.5 * test_data['ex']))

# 运行加速后的选择过程
sup_wBH = local_utils.weighted_BH(sup_calib_scores, sup_calib_weights, sup_test_scores, sup_test_weights, q = q)
sup_ , sup_wCC_hete, sup_wCC_homo, sup_wCC_dtm = local_utils.weighted_CS(sup_calib_scores, sup_calib_weights, sup_test_scores, sup_test_weights, q = q)

truth_test = 1 - np.array(test_data['class'])
sup_wBH_eval = eval_FDR(sup_wBH, truth_test)
sup_wCC_hete_eval = eval_FDR(sup_wCC_hete, truth_test)
sup_wCC_homo_eval = eval_FDR(sup_wCC_homo, truth_test)
sup_wCC_dtm_eval = eval_FDR(sup_wCC_dtm, truth_test)


# =============================================================================
# # 2. 条件化分类方法 (Conditional Approach)
# =============================================================================
print("⚡ 正在解算条件化方法得分...")
calib_class_mask = (calib_data['class'] == 0).values
cond_calib_scores = sup_calib_scores[calib_class_mask]
cond_calib_weights = sup_calib_weights[calib_class_mask]

cond_wBH = local_utils.weighted_BH(cond_calib_scores, cond_calib_weights, sup_test_scores, sup_test_weights, q = q)
cond_, cond_wCC_hete, cond_wCC_homo, cond_wCC_dtm = local_utils.weighted_CS(cond_calib_scores, cond_calib_weights, sup_test_scores, sup_test_weights, q = q)

cond_wBH_eval = eval_FDR(cond_wBH, truth_test)
cond_wCC_hete_eval = eval_FDR(cond_wCC_hete, truth_test)
cond_wCC_homo_eval = eval_FDR(cond_wCC_homo, truth_test)
cond_wCC_dtm_eval = eval_FDR(cond_wCC_dtm, truth_test)


# =============================================================================
# # 3. 异常检测方法 (One-Class SVM Outlier Detection)
# =============================================================================
print("⚡ 正在训练一分类异常检测模型 (OneClassSVM)...")
out_mdl = svm.OneClassSVM(kernel="rbf", gamma=0.01)
train_class_mask = train_data['class'] == 0
out_mdl.fit(train_data[train_class_mask].iloc[:,0:62])

out_calib_scores = out_mdl.score_samples(calib_data[calib_class_mask].iloc[:,0:62])
out_test_scores = out_mdl.score_samples(test_data.iloc[:,0:62])

print("⚡ [矩阵计算启动] 正在以火箭速度解算所有 WBH 与 WCS 变体...")
out_wBH = local_utils.weighted_BH(out_calib_scores, cond_calib_weights, out_test_scores, sup_test_weights, q = q)
out_wBH_eval = eval_FDR(out_wBH, truth_test)
print(out_wBH_eval)
 
out_, out_wCC_hete, out_wCC_homo, out_wCC_dtm = local_utils.weighted_CS(out_calib_scores, cond_calib_weights, out_test_scores, sup_test_weights, q = q)

out_wCC_hete_eval = eval_FDR(out_wCC_hete, truth_test)
out_wCC_homo_eval = eval_FDR(out_wCC_homo, truth_test)
out_wCC_dtm_eval = eval_FDR(out_wCC_dtm, truth_test)
     

# =============================================================================
# # 4. 汇总导出数据
# =============================================================================
results = pd.DataFrame(np.array((sup_wBH_eval, sup_wCC_hete_eval, sup_wCC_homo_eval, sup_wCC_dtm_eval, 
                                 cond_wBH_eval, cond_wCC_hete_eval, cond_wCC_homo_eval, cond_wCC_dtm_eval,
                                 out_wBH_eval, out_wCC_hete_eval, out_wCC_homo_eval, out_wCC_dtm_eval)))
results.columns = ["nrej", "fdp", "power"]
results['method'] = ["WBH", "WCS.hete", "WCS.homo", "WCS.dtm"] * 3
results['setting'] = ["sup_class"]*4 + ["cond_class"]*4 + ["outlier"]*4

# 快速计算集合差异
results['ndiff1'] = [0, len(set(sup_wBH)-set(sup_wCC_hete)), len(set(sup_wBH)-set(sup_wCC_homo)), len(set(sup_wBH)-set(sup_wCC_dtm)),
                     0, len(set(cond_wBH)-set(cond_wCC_hete)), len(set(cond_wBH)-set(cond_wCC_homo)), len(set(cond_wBH)-set(cond_wCC_dtm)),
                     0, len(set(out_wBH)-set(out_wCC_hete)), len(set(out_wBH)-set(out_wCC_homo)), len(set(out_wBH)-set(out_wCC_dtm))]

results['ndiff2'] = [0, len(set(sup_wCC_hete)-set(sup_wBH)), len(set(sup_wCC_homo)-set(sup_wBH)), len(set(sup_wCC_dtm)-set(sup_wBH)),
                     0, len(set(cond_wCC_hete)-set(cond_wBH)), len(set(cond_wCC_homo)-set(cond_wBH)), len(set(cond_wCC_dtm)-set(cond_wBH)),
                     0, len(set(out_wCC_hete)-set(out_wBH)), len(set(out_wCC_homo)-set(out_wBH)), len(set(out_wCC_dtm)-set(out_wBH))]

results['ntest'] = np.sum(in_test)
results['seed'] = seed
 
save_path = "./results"
os.makedirs(save_path, exist_ok=True)
results.to_csv(f"{save_path}/seed_{seed}_q_{q}.csv", index=False)

print(f"--- 流程全部顺利结束！结果成功存入: {save_path}/seed_{seed}_q_{q}.csv ---")