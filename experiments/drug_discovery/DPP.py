# =============================================================================
# # 修复核心：将官方工具包与本地统计工具包做严格重命名区分，防止互相覆盖
# =============================================================================
from DeepPurpose import utils as dp_utils
from DeepPurpose import dataset, CompoundPred
import warnings
import os
import shutil
import urllib.request
import numpy as np
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# 导入本地统计函数包
import utils as local_utils
from utils import eval_sel

warnings.filterwarnings("ignore")

# 屏蔽 RDKit 在提取 Morgan 指纹时输出的冗余警告
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


# 【劫持覆盖】用高度精确化的矩阵函数，原地替换掉本地导入的原版慢速函数
local_utils.weighted_BH = fast_weighted_BH
local_utils.weighted_CS = fast_weighted_CS
print("🚀 [系统提示] 成功注入【精准等价修复版】NumPy 矩阵加速引擎！")


# =============================================================================
# # 🛡️ 【数据与工程防错】标准 PyTorch 数据加载管道
# =============================================================================
class DeepPurposeDataset(Dataset):
    def __init__(self, df_data):
        self.df = df_data.reset_index(drop=True)
        self.v_d = np.stack(self.df['drug_encoding'].values)
        self.labels = self.df['Label'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            'v_d': torch.tensor(self.v_d[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }


def safe_pytorch_predict(dp_model, processed_data):
    net = dp_model.model
    net.eval()
    
    torch_dataset = DeepPurposeDataset(processed_data)
    loader = DataLoader(torch_dataset, batch_size=2048, shuffle=False, num_workers=0)
    
    preds = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    
    print(f"原生 PyTorch 正在单线程高效预测 {len(torch_dataset)} 个化合物样本...")
    with torch.no_grad():
        for batch in loader:
            inputs = batch['v_d'].to(device)
            try:
                score = net(inputs)
            except Exception:
                score = net(batch)
                
            if isinstance(score, tuple):
                score = score[0]
            preds.extend(score.cpu().numpy().flatten())
            
    return preds


def ensure_hiv_dataset(path="./data"):
    """大小写免疫安全版：只要本地有 HIV 数据集就不再强行复制，直接放行"""
    os.makedirs(path, exist_ok=True)
    lower_path = os.path.join(path, "hiv.csv")
    upper_path = os.path.join(path, "HIV.csv")

    # 1. 如果这两个路径中有任何一个在物理上存在，说明数据已经在本地了，直接返回
    if os.path.exists(lower_path) or os.path.exists(upper_path):
        print("✅ [系统提示] 检测到本地已存在 HIV 数据集，跳过下载与强行复制。")
        return

    # 2. 如果本地一个都没有，再去尝试下载
    candidates = ["https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv"]
    last_error = None
    for url in candidates:
        try:
            print(f"📡 正在从网络下载 HIV 数据集: {url} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as response, open(upper_path, "wb") as out:
                out.write(response.read())
            print("💾 下载成功！")
            return
        except Exception as err:
            last_error = err

    raise RuntimeError(f"Failed to prepare HIV dataset. Last error: {last_error}")


# =============================================================================
# # 主程序核心业务逻辑
# =============================================================================
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
q = (int(sys.argv[2]) / 10) if len(sys.argv) > 2 else 0.5

ensure_hiv_dataset(path="./data")
X_drugs, y, drugs_index = dataset.load_HIV(path="./data")
drug_encoding = 'Morgan'

np.random.seed(seed)
n = len(y)
reind = np.random.permutation(n)

split_idx = int(n * 0.4 + 1)
X_drugs_train, y_train = X_drugs[reind[:split_idx]], y[reind[:split_idx]]
X_drugs_other, y_other = X_drugs[reind[split_idx:]], y[reind[split_idx:]]

ttrain, tval, ttest = dp_utils.data_process(X_drug = X_drugs_train, y = y_train,
                                           drug_encoding = drug_encoding,
                                           split_method='random', frac=[0.7,0.1,0.2],
                                           random_seed = seed)

config = dp_utils.generate_config(drug_encoding = drug_encoding,
                                 cls_hidden_dims = [1024,1024,512],
                                 train_epoch = 3,
                                 LR = 0.001,
                                 batch_size = 128,
                                 hidden_dim_drug = 128,
                                 mpnn_hidden_size = 128,
                                 mpnn_depth = 3)

model = CompoundPred.model_initialize(**config)
model.train(ttrain, tval, ttest)

if torch.cuda.is_available():
    model.model = model.model.to('cuda:0')

dother = dp_utils.data_process(X_drug = X_drugs_other, y = y_other,
                              drug_encoding = drug_encoding,
                              split_method='no_split',
                              random_seed = seed)

print("--- 开始安全预测 dother ---")
all_pred = np.array(safe_pytorch_predict(model, dother))

print("--- 开始安全预测 ttrain ---")
train_pred = np.array(safe_pytorch_predict(model, ttrain))

print("--- 预测全部顺利通过！开始执行统计切片 ---")

eta = all_pred - np.mean(train_pred)
p_x = np.clip(np.exp(eta) / (1 + np.exp(eta)), 1e-9, 0.8)
in_calib = np.random.binomial(1, p_x, size=len(p_x))

if in_calib.sum() == 0:
    in_calib[np.argmax(p_x)] = 1
elif in_calib.sum() == len(in_calib):
    in_calib[np.argmin(p_x)] = 0

calib_mask = (in_calib == 1)
test_mask = ~calib_mask

dcalib = dp_utils.data_process(X_drug = X_drugs_other[calib_mask], y = y_other[calib_mask],
                              drug_encoding = drug_encoding, split_method='no_split', random_seed = seed)
dtest = dp_utils.data_process(X_drug = X_drugs_other[test_mask], y = y_other[test_mask],
                             drug_encoding = drug_encoding, split_method='no_split', random_seed = seed)

hat_mu_calib = all_pred[calib_mask]
hat_mu_test = all_pred[test_mask]
y_calib = np.array(dcalib["Label"])
y_test = np.array(dtest['Label'])

w_calib = np.array(1 / p_x[calib_mask] - 1)
w_test = np.array(1 / p_x[test_mask] - 1)

c = 0
calib_scores_res = y_calib - hat_mu_calib
calib_scores_sub = - hat_mu_calib
calib_scores_clip = 100 * (y_calib > c) + c * (y_calib <= c) - hat_mu_calib
test_scores = c - hat_mu_test

print("⚡ [矩阵计算启动] 正在以火箭速度解算 WBH 与 WCS 变体...")

# 这里的传参切片映射已被完全拉齐
BH_res = local_utils.weighted_BH(calib_scores_res, w_calib, test_scores, w_test, q)
BH_sub = local_utils.weighted_BH(calib_scores_sub[y_calib <= c], w_calib[y_calib <= c], test_scores, w_test, q)
BH_clip = local_utils.weighted_BH(calib_scores_clip, w_calib, test_scores, w_test, q)

WCS_res_0, WCS_res_hete, WCS_res_homo, WCS_res_dtm = local_utils.weighted_CS(calib_scores_res, w_calib, test_scores, w_test, q)
WCS_sub_0, WCS_sub_hete, WCS_sub_homo, WCS_sub_dtm = local_utils.weighted_CS(calib_scores_sub[y_calib <= c], w_calib[y_calib <= c], test_scores, w_test, q)
WCS_clip_0, WCS_clip_hete, WCS_clip_homo, WCS_clip_dtm = local_utils.weighted_CS(calib_scores_clip, w_calib, test_scores, w_test, q)

# 4. 指标计算与最终数据导出
BH_res_fdp, BH_res_power = eval_sel(BH_res, y_test, np.array([c]*len(y_test)))
BH_sub_fdp, BH_sub_power = eval_sel(BH_sub, y_test, np.array([c]*len(y_test)))
BH_clip_fdp, BH_clip_power = eval_sel(BH_clip, y_test, np.array([c]*len(y_test)))

all_BH = [BH_res, BH_sub, BH_clip]
all_sel = [[WCS_res_hete, WCS_res_homo, WCS_res_dtm],
           [WCS_sub_hete, WCS_sub_homo, WCS_sub_dtm],
           [WCS_clip_hete, WCS_clip_homo, WCS_clip_dtm]]
fdp = [BH_res_fdp, BH_sub_fdp, BH_clip_fdp]
power = [BH_res_power, BH_sub_power, BH_clip_power]
ndiff = [0] * 3
nsel = [len(BH_res), len(BH_sub), len(BH_clip)]
nsame = [len(BH_res), len(BH_sub), len(BH_clip)]

for ii in range(3):
    sels = all_sel[ii]
    tpowers, tfdps, tnsels, tndiffs, tnsames = [], [], [], [], []
    for jj in range(3):
        tfdp, tpower = eval_sel(sels[jj], y_test, np.array([c]*len(y_test)))
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

res = pd.DataFrame({
    "FDP": fdp, "power": power, "nsel": nsel, "ndiff": ndiff, "nsame": nsame,
    "score": ["res", "sub", "clip"] + ["res"]*3 + ["sub"]*3 + ["clip"]*3,
    "method": ["WBH"]*3 + ['WCS.hete', 'WCS.homo', "WCS.dtm"] *3,
    "q": q, "seed": seed
})

save_path = "./DPP_results"
os.makedirs(save_path, exist_ok=True)
res.to_csv(os.path.join(save_path, f"seed{seed}q{q}.csv"), index=False)

print(f"--- 流程全部顺利结束！结果成功存入: {save_path}/seed{seed}q{q}.csv ---")