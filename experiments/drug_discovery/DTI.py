from DeepPurpose import utils as dp_utils
from DeepPurpose import dataset, CompoundPred
import warnings
import numpy as np
import random 
import pandas as pd
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from DeepPurpose import DTI as models

# 导入本地统计函数包
import utils as local_utils
from utils import weighted_BH, weighted_CS, eval_sel

warnings.filterwarnings("ignore")

# 屏蔽 RDKit 冗余警告
try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.warning")
except Exception:
    pass

# =============================================================================
# # 🚀 【数学等价修复版】NumPy 矩阵广播加速引擎（无缝同步原始算法的所有统计机制）
# =============================================================================
def fast_weighted_BH(calib_scores, calib_weights, test_scores, test_weights, q=0.1):
    calib_scores = np.array(calib_scores)
    calib_weights = np.array(calib_weights)
    test_scores = np.array(test_scores)
    test_weights = np.array(test_weights)
    
    n_calib = len(calib_scores)
    n_test = len(test_scores)
    sum_calib_weight = np.sum(calib_weights)
    
    # 构造联合数组，严格复现 pd.concat 的合并序列排序
    calib_part = np.stack([calib_scores, calib_weights, np.ones(n_calib), np.arange(n_calib)], axis=1)
    test_part = np.stack([test_scores, test_weights, np.zeros(n_test), np.arange(n_test)], axis=1)
    combined = np.concatenate([calib_part, test_part], axis=0)
    
    # 指定 kind='mergesort'（稳定排序），确保平局样本在 NumPy 和 Pandas 下行为百分百一致
    sort_idx = np.argsort(combined[:, 0], kind='mergesort')
    combined_sorted = combined[sort_idx]
    
    # 计算排在当前样本前的 [校准样本权重] 累加和
    calib_w_indicators = combined_sorted[:, 1] * combined_sorted[:, 2]
    cum_calib_weights = np.cumsum(calib_w_indicators)
    
    #np.random.seed(None)  # 保持运行时的真实扰动
    rand_vals = np.random.uniform(size=n_test)
    
    p_vals_combined = np.full(len(combined), -1.0)
    test_mask_sorted = (combined_sorted[:, 2] == 0)
    test_indices_in_sorted = np.where(test_mask_sorted)[0]
    
    prev_calib_w = np.zeros(n_test)
    valid_prev_mask = test_indices_in_sorted > 0
    prev_calib_w[valid_prev_mask] = cum_calib_weights[test_indices_in_sorted[valid_prev_mask] - 1]
    
    t_weights = combined_sorted[test_indices_in_sorted, 1]
    computed_pvals = (prev_calib_w + t_weights * rand_vals) / (sum_calib_weight + t_weights)
    
    p_vals_combined[test_indices_in_sorted] = computed_pvals
    
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
    
    # 高维广播矩阵化处理：一次性解析 calib_scores < test_scores[k]
    scores_calib_less_test = calib_scores[:, np.newaxis] < test_scores[np.newaxis, :]
    sum_w_calib_less_test = np.sum(scores_calib_less_test * calib_weights[:, np.newaxis], axis=0)
    
    # 针对非对称留一法中 test_scores[j] < test_scores[k] 展开交互计算
    matrix_j_less_k = test_scores[:, np.newaxis] < test_scores[np.newaxis, :]
    
    # 求解包含等号平局概率的 w_pvals
    calib_less_self = calib_scores[:, np.newaxis] < test_scores[np.newaxis, :]
    calib_equal_self = calib_scores[:, np.newaxis] == test_scores[np.newaxis, :]
    sum_calib_less_self = np.sum(calib_less_self * calib_weights[:, np.newaxis], axis=0)
    sum_calib_equal_self = np.sum(calib_equal_self * calib_weights[:, np.newaxis], axis=0)
    
    #np.random.seed(None)
    rand_w = np.random.uniform(size=ntest)
    w_pvals = (sum_calib_less_self + (sum_calib_equal_self + test_weights) * rand_w) / (sum_calib_weight + test_weights)
    
    # 计算内部虚拟拒绝空间
    Rj_sizes = np.zeros(ntest)
    threshold_line = q * np.linspace(1, ntest, num=ntest) / ntest
    
    for j in range(ntest):
        pval_j = sum_w_calib_less_test + test_weights * matrix_j_less_k[j, :]
        pval_j[j] = 0.0  # 挖空样本自身
        
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
    
    # 三重同质/异质统计剪裁
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

# 劫持方法
local_utils.weighted_BH = fast_weighted_BH
local_utils.weighted_CS = fast_weighted_CS
print("🚀 [系统提示] WBH 与 WCS 矩阵加速引擎注入成功！")


# =============================================================================
# # 🛡️ 【黑魔法 2】标准 PyTorch DTI 数据管道（防死锁、防 KeyError）
# =============================================================================
class DTIPyTorchDataset(Dataset):
    """
    针对 DTI 任务的 PyTorch Dataset 封装，同时提取药物和靶点特征。
    将数据预先转为 NumPy 矩阵，阻断 Pandas 键值错误。
    """
    def __init__(self, df_data):
        self.df = df_data.reset_index(drop=True)
        self.v_d = np.stack(self.df['drug_encoding'].values)
        self.v_p = np.stack(self.df['target_encoding'].values)
        self.labels = self.df['Label'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            'v_d': torch.tensor(self.v_d[idx], dtype=torch.float32),
            'v_p': torch.tensor(self.v_p[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }


def safe_dti_predict(dp_model, processed_data):
    """单线程批量前向传播，彻底防死锁并填满 GPU 算力"""
    net = dp_model.model
    net.eval()
    
    torch_dataset = DTIPyTorchDataset(processed_data)
    loader = DataLoader(torch_dataset, batch_size=2048, shuffle=False, num_workers=0)
    
    preds = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    
    print(f"原生 PyTorch 正在高效推理 {len(torch_dataset)} 个 DTI 样本...")
    with torch.no_grad():
        for batch in loader:
            inputs_d = batch['v_d'].to(device)
            inputs_p = batch['v_p'].to(device)
            
            # 兼容不同 DeepPurpose 底层网络的前向传播调用方式
            try:
                score = net(inputs_d, inputs_p)
            except Exception:
                score = net(batch)
                
            if isinstance(score, tuple):
                score = score[0]
            preds.extend(score.cpu().numpy().flatten())
            
    return preds


# =============================================================================
# # 数据加载与参数设置
# =============================================================================
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
q = (int(sys.argv[2]) / 10) if len(sys.argv) > 2 else 0.5
qpop = int(sys.argv[3]) if len(sys.argv) > 3 else 5

X_drugs, X_targets, y = dataset.load_process_DAVIS(path = './data', binary = False, convert_to_log = True, threshold = 30) 
drug_encoding, target_encoding = 'Morgan', 'Conjoint_triad' 

n = len(y) 
np.random.seed(seed)
reind = np.random.permutation(n)

# 划分 Train 与 Other
split_idx = int(n * 0.2 + 1)
X_drugs_train, X_targets_train, y_train = X_drugs[reind[:split_idx]], X_targets[reind[:split_idx]], y[reind[:split_idx]]
X_drugs_other, X_targets_other, y_other = X_drugs[reind[split_idx:]], X_targets[reind[split_idx:]], y[reind[split_idx:]]

# 训练集编码
ttrain, tval, ttest = dp_utils.data_process(X_drugs_train, X_targets_train, y_train, 
                                           drug_encoding, target_encoding, 
                                           split_method='random', frac=[0.7,0.1,0.2],
                                           random_seed = seed) 

config = dp_utils.generate_config(drug_encoding = drug_encoding, 
                                  target_encoding = target_encoding, 
                                  cls_hidden_dims = [1024,1024,512], 
                                  train_epoch = 10, 
                                  LR = 0.001, 
                                  batch_size = 128,
                                  hidden_dim_drug = 128,
                                  mpnn_hidden_size = 128,
                                  mpnn_depth = 3, 
                                  cnn_target_filters = [32,64,96],
                                  cnn_target_kernels = [4,8,12])

model = models.model_initialize(**config)
model.train(ttrain, tval, ttest)

# 推送到 GPU
if torch.cuda.is_available():
    model.model = model.model.to('cuda:0')

# Other 集合编码
dother = dp_utils.data_process(X_drugs_other, X_targets_other, y_other, 
                              drug_encoding, target_encoding, 
                              split_method='no_split', random_seed = seed) 

# 使用安全预测函数替换原始的 model.predict()
print("--- 开始安全预测 dother ---")
all_pred = np.array(safe_dti_predict(model, dother))
print("--- 开始安全预测 ttrain ---")
train_pred = np.array(safe_dti_predict(model, ttrain))

# 计算倾向性得分并抽取校准集
p_x = np.exp(2*(all_pred - np.mean(train_pred))) / (1+np.exp(2*(all_pred - np.mean(train_pred))))
in_calib = np.random.binomial(1, p_x, size=len(p_x))

if in_calib.sum() == 0:
    in_calib[np.argmax(p_x)] = 1
elif in_calib.sum() == len(in_calib):
    in_calib[np.argmin(p_x)] = 0

calib_mask = (in_calib == 1)
test_mask = ~calib_mask

# 【安全规避】：绕开 Pandas 切片索引缺陷，先用 NumPy 掩码过滤原始数组，重新做 data_process
dcalib = dp_utils.data_process(X_drugs_other[calib_mask], X_targets_other[calib_mask], y_other[calib_mask],
                              drug_encoding, target_encoding, split_method='no_split', random_seed = seed)
dtest = dp_utils.data_process(X_drugs_other[test_mask], X_targets_other[test_mask], y_other[test_mask],
                             drug_encoding, target_encoding, split_method='no_split', random_seed = seed)


# =============================================================================
# # 🚀 【黑魔法 3】字典映射快速分位数计算（干掉原本极慢的 Pandas 重复遍历）
# =============================================================================
print("⚡ 正在优化计算序列分位数...")
# 建立全量训练集的快速查找字典
train_df = pd.DataFrame({'Seq': ttrain['Target Sequence'], 'Label': ttrain['Label']})
seq_to_labels = train_df.groupby('Seq')['Label'].apply(list).to_dict()
global_labels = ttrain['Label'].values

def get_fast_quantiles(df_target):
    """通过哈希字典映射，批量极速求解序列的分位数"""
    q_lists = {2: [], 5: [], 7: [], 8: [], 9: []}
    for seq in df_target['Target Sequence']:
        labels = seq_to_labels.get(seq, global_labels)
        for q_int in [2, 5, 7, 8, 9]:
            q_lists[q_int].append(np.quantile(labels, q_int / 10.0))
    return [np.array(q_lists[k]) for k in [2, 5, 7, 8, 9]]

testq2, testq5, testq7, testq8, testq9 = get_fast_quantiles(dtest)
calibq2, calibq5, calibq7, calibq8, calibq9 = get_fast_quantiles(dcalib)


# =============================================================================
# # 构建数据矩阵与下采样
# =============================================================================
# 重新用安全推理得到对齐的预测值
hat_mu_calib = np.array(safe_dti_predict(model, dcalib))
hat_mu_test = np.array(safe_dti_predict(model, dtest))

y_calib = np.array(dcalib["Label"])
w_calib = np.array(1/p_x[calib_mask] - 1)
y_test = np.array(dtest['Label'])
w_test = np.array(1/p_x[test_mask] - 1)

data_calib = pd.DataFrame({"calib_pred": hat_mu_calib, "calib_true": y_calib, "calib_w": w_calib,
                          "q2": calibq2, "q5": calibq5, "q7": calibq7, "q8": calibq8, "q9": calibq9})
data_test = pd.DataFrame({"test_pred": hat_mu_test, "test_true": y_test, "test_w": w_test,
                          "q2": testq2, "q5": testq5, "q7": testq7, "q8": testq8, "q9": testq9})

# 限制测试集样本数目为 5000 
if data_test.shape[0] > 5000:
    test_sub = np.random.permutation(data_test.shape[0])[0:5000]
    data_test = data_test.iloc[test_sub,:].reset_index(drop=True)

# 重新提炼数组进行数学统计
hat_mu_test = data_test['test_pred'].values
y_test = data_test['test_true'].values
w_test = data_test['test_w'].values

cname = 'q' + str(int(qpop))
c_calib = data_calib[cname].values
c_test = data_test[cname].values

calib_scores_res = y_calib - hat_mu_calib 
calib_scores_clip = 100 * (y_calib > c_calib) + c_calib * (y_calib <= c_calib) - hat_mu_calib
test_scores = c_test - hat_mu_test


# =============================================================================
# # 运行多重假设检验程序（此时已被矩阵化劫持函数覆盖，瞬时秒过）
# =============================================================================
print("⚡ [矩阵计算启动] 正在高速解算 WBH 与 WCS 变体...")

BH_res = local_utils.weighted_BH(calib_scores_res, w_calib, test_scores, w_test, q)   
BH_clip = local_utils.weighted_BH(calib_scores_clip, w_calib, test_scores, w_test, q)

BH_res_fdp, BH_res_power = eval_sel(BH_res, y_test, c_test)
BH_clip_fdp, BH_clip_power = eval_sel(BH_clip, y_test, c_test)

CS_res_0, CS_res_hete, CS_res_homo, CS_res_dtm = local_utils.weighted_CS(calib_scores_res, w_calib, test_scores, w_test, q) 
CS_clip_0, CS_clip_hete, CS_clip_homo, CS_clip_dtm = local_utils.weighted_CS(calib_scores_clip, w_calib, test_scores, w_test, q)

# =============================================================================
# # 汇总与数据落地
# =============================================================================
all_BH = [BH_res, BH_clip]
all_sel = [[CS_res_hete, CS_res_homo, CS_res_dtm], [CS_clip_hete, CS_clip_homo, CS_clip_dtm]]
fdp = [BH_res_fdp, BH_clip_fdp]
power = [BH_res_power, BH_clip_power] 
ndiff = [0] * 2
nsel = [len(BH_res), len(BH_clip)]
nsame = [len(BH_res), len(BH_clip)]

for ii in range(2):
    sels = all_sel[ii]
    tpowers, tfdps, tnsels, tndiffs, tnsames = [], [], [], [], []
    for jj in range(3):
        tfdp, tpower = eval_sel(sels[jj], y_test, c_test)
        tpowers.append(tpower)
        tfdps.append(tfdp)
        tnsels.append(len(sels[jj]))
        tndiffs.append(len(np.setxor1d(all_BH[ii], sels[jj])))
        tnsames.append(len(np.intersect1d(all_BH[ii], sels[jj])))
    fdp += tfdps
    power += tpowers
    ndiff += tndiffs
    nsel += tnsels
    nsame += tnsames
 
res = pd.DataFrame({"FDP": fdp, "power": power, "nsel": nsel, "ndiff": ndiff, "nsame": nsame,
                    "score": ["res", "clip"] + ["res"]*3 + ["clip"]*3,
                    "method": ["WBH"]*2 + ['WCS.hete', 'WCS.homo', "WCS.dtm"] *2,
                    "q": q, "seed": seed, "qpop": qpop})

save_path = "./DTI_results"
os.makedirs(save_path, exist_ok=True)
res.to_csv(f"{save_path}/seed{seed}q{q}qpop{qpop}.csv", index=False)

print(f"--- 流程全部顺利结束！结果成功存入: {save_path}/seed{seed}q{q}qpop{qpop}.csv ---")