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
# # 🚀 【核心修复】数学等价矩阵化加速引擎（100% 还原 utils.py 算法逻辑）
# =============================================================================
def fast_weighted_BH(calib_scores, calib_weights, test_scores, test_weights, q=0.1):
    calib_scores = np.array(calib_scores)
    calib_weights = np.array(calib_weights)
    test_scores = np.array(test_scores)
    test_weights = np.array(test_weights)
    
    n_calib = len(calib_scores)
    n_test = len(test_scores)
    sum_calib_weight = np.sum(calib_weights)
    
    # 构造联合数组，精准对齐原作者 pd.concat 合并序列的排序与打散基准
    calib_part = np.stack([calib_scores, calib_weights, np.ones(n_calib), np.arange(n_calib)], axis=1)
    test_part = np.stack([test_scores, test_weights, np.zeros(n_test), np.arange(n_test)], axis=1)
    combined = np.concatenate([calib_part, test_part], axis=0)
    
    # 强制指定 kind='mergesort'（稳定排序），完美拉齐 Pandas 默认平局连接行为
    sort_idx = np.argsort(combined[:, 0], kind='mergesort')
    combined_sorted = combined[sort_idx]
    
    # 累加排在当前位置之前的【校准样本权重】
    calib_w_indicators = combined_sorted[:, 1] * combined_sorted[:, 2]
    cum_calib_weights = np.cumsum(calib_w_indicators)
    
    #np.random.seed(None)  # 保持运行时的真实随机性
    rand_vals = np.random.uniform(size=n_test)
    
    test_mask_sorted = (combined_sorted[:, 2] == 0)
    test_indices_in_sorted = np.where(test_mask_sorted)[0]
    
    prev_calib_w = np.zeros(n_test)
    valid_prev_mask = test_indices_in_sorted > 0
    prev_calib_w[valid_prev_mask] = cum_calib_weights[test_indices_in_sorted[valid_prev_mask] - 1]
    
    t_weights = combined_sorted[test_indices_in_sorted, 1]
    computed_pvals = (prev_calib_w + t_weights * rand_vals) / (sum_calib_weight + t_weights)
    
    original_test_pvals = np.zeros(n_test)
    original_test_ids = combined_sorted[test_indices_in_sorted, 3].astype(int)
    original_test_pvals[original_test_ids] = computed_pvals
    
    df_test = pd.DataFrame({"id": np.arange(n_test), "pvals": original_test_pvals})
    df_test_sorted = df_test.sort_values(by='pvals', kind='mergesort').reset_index(drop=True)
    
    df_test_sorted['threshold'] = q * np.linspace(1, n_test, num=n_test) / n_test 
    idx_smaller = [j for j in range(n_test) if df_test_sorted.iloc[j, 1] <= df_test_sorted.iloc[j, 2]]
    
    if len(idx_smaller) == 0:
        return np.array([])
    else:
        return np.array(df_test_sorted['id'].iloc[range(np.max(idx_smaller) + 1)])


def fast_weighted_CS(calib_scores, calib_weights, test_scores, test_weights, q=0.1):
    calib_scores = np.array(calib_scores)
    calib_weights = np.array(calib_weights)
    test_scores = np.array(test_scores)
    test_weights = np.array(test_weights)
    
    sum_calib_weight = np.sum(calib_weights)
    ntest = len(test_scores)
    
    # 彻底击碎嵌套循环：利用 NumPy 广播机制一气呵成算齐矩阵
    # 1. 校准集与测试集的两两对比 (n_calib, n_test)
    scores_calib_less_test = calib_scores[:, np.newaxis] < test_scores[np.newaxis, :]
    sum_w_calib_less_test = np.sum(scores_calib_less_test * calib_weights[:, np.newaxis], axis=0)
    
    # 2. 测试集两两交叉的交互矩阵 (ntest, ntest)
    matrix_j_less_k = test_scores[:, np.newaxis] < test_scores[np.newaxis, :]
    
    # 3. 用于计算包含平局和小数波动的 w_pvals 矩阵
    calib_equal_self = calib_scores[:, np.newaxis] == test_scores[np.newaxis, :]
    sum_calib_equal_self = np.sum(calib_equal_self * calib_weights[:, np.newaxis], axis=0)
    
    #np.random.seed(None)
    rand_w = np.random.uniform(size=ntest)
    w_pvals = (sum_w_calib_less_test + (sum_calib_equal_self + test_weights) * rand_w) / (sum_calib_weight + test_weights)
    
    # 精准映射留一法（Leave-one-out）内部虚拟拒绝集合大小 Rj
    Rj_sizes = np.zeros(ntest)
    threshold_line = q * np.linspace(1, ntest, num=ntest) / ntest
    
    for j in range(ntest):
        # 对应 pval_j[k] = np.sum(calib_weights[calib_scores < test_scores[k]]) + test_weights[k] * (test_scores[j] < test_scores[k])
        pval_j = sum_w_calib_less_test + test_weights * matrix_j_less_k[j, :]
        pval_j[j] = 0.0  # 还原 k != j 的挖空设定
        
        sort_p_idx = np.argsort(pval_j, kind='mergesort')
        pval_j_sorted = pval_j[sort_p_idx] / (sum_calib_weight + test_weights[j])
        
        idx_small_j = np.where(pval_j_sorted <= threshold_line)[0]
        if len(idx_small_j) > 0:
            Rj_sizes[j] = np.max(idx_small_j) + 1
            
    Cj = q * Rj_sizes / ntest
    xis = np.random.uniform(size=ntest)
    rand_homo = np.random.uniform(size=1)[0]
    
    df_all = pd.DataFrame({
        "id": range(ntest), "pval": w_pvals, "c": Cj, 
        "hete_Rj": Rj_sizes * xis, 
        "homo_Rj": Rj_sizes * rand_homo, 
        "Rj": Rj_sizes
    })
    
    pj_sel0 = w_pvals[w_pvals <= Cj]
    if len(pj_sel0) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]) 
    
    valid_mask = df_all['pval'] <= df_all['c']
    
    # 三重随机/确定性统计剪裁（Pruning）还原
    df_hete = df_all[valid_mask].sort_values(by='hete_Rj', kind='mergesort')
    smaller_hete = np.where(df_hete['hete_Rj'].values <= np.linspace(1, len(df_hete), num=len(df_hete)))[0]
    idx_sel_hete = np.array(df_hete['id'].iloc[:np.max(smaller_hete) + 1]) if len(smaller_hete) > 0 else np.array([])
    
    df_homo = df_all[valid_mask].sort_values(by='homo_Rj', kind='mergesort')
    smaller_homo = np.where(df_homo['homo_Rj'].values <= np.linspace(1, len(df_homo), num=len(df_homo)))[0]
    idx_sel_homo = np.array(df_homo['id'].iloc[:np.max(smaller_homo) + 1]) if len(smaller_homo) > 0 else np.array([])
    
    df_dete = df_all[valid_mask].sort_values(by='homo_Rj', kind='mergesort')
    smaller_dete = np.where(df_dete['Rj'].values <= np.linspace(1, len(df_dete), num=len(df_dete)))[0]
    idx_sel_dete = np.array(df_dete['id'].iloc[:np.max(smaller_dete) + 1]) if len(smaller_dete) > 0 else np.array([])
    
    return np.array(df_dete['id']), idx_sel_hete, idx_sel_homo, idx_sel_dete

# 强行劫持函数，作废慢速版本
local_utils.weighted_BH = fast_weighted_BH
local_utils.weighted_CS = fast_weighted_CS
print("🚀 [系统提示] 【精准对齐修复版】WBH/WCS 矩阵加速引擎已成功注入！")


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
train_data = remain_data.iloc[(ncalib+1):,].reset_index(drop=True)


# =============================================================================
# # 1. 经典联合方法 (Super-population Approach)
# =============================================================================
print("⚡ 正在训练支持向量机分类模型 (SVC)...")
sup_mdl = svm.SVC(kernel="rbf", gamma=0.01, probability=True).fit(train_data.iloc[:,0:62], train_data['class'])

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