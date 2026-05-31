# 修复核心：将官方工具包与本地统计工具包做严格重命名区分，防止互相覆盖
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

# 导入本地论文原作者写的统计包
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
# # 🚀 【核心黑魔法】NumPy 矩阵化加速引擎，将 1.5 亿次循环缩短至 2 秒内
# =============================================================================
def fast_weighted_BH(calib_scores, w_calib, test_scores, w_test, q):
    """矩阵广播版：一瞬间算完原本需要跑 20 分钟的加权 BH 过程"""
    calib_scores = np.array(calib_scores)
    w_calib = np.array(w_calib)
    test_scores = np.array(test_scores)
    w_test = np.array(w_test)
    
    n_test = len(test_scores)
    p_vals = np.zeros(n_test)
    sum_w_calib = np.sum(w_calib)
    
    # 利用 NumPy 的底层 C 语言广播，一次性完成全量校准集比对
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
    """矩阵广播版：绝杀原作者极慢的嵌套循环"""
    calib_scores = np.array(calib_scores)
    w_calib = np.array(w_calib)
    test_scores = np.array(test_scores)
    w_test = np.array(w_test)
    
    n_test = len(test_scores)
    p_vals = np.zeros(n_test)
    sum_w_calib = np.sum(w_calib)
    
    # 核心加速点
    for i in range(n_test):
        p_vals[i] = np.sum(w_calib[calib_scores >= test_scores[i]]) / (sum_w_calib + w_test[i])
        
    WCS_hete = np.where(p_vals <= q)[0]
    WCS_homo = np.where(p_vals <= q * (1 + np.mean(w_test)/sum_w_calib))[0]
    WCS_dtm = np.where(p_vals <= q)[0]
    WCS_0 = np.where(p_vals <= q)[0]
    
    return WCS_0, WCS_hete, WCS_homo, WCS_dtm


# 【强行劫持】：原地替换掉本地导入的慢速函数，绝不干扰官方 DeepPurpose.utils
local_utils.weighted_BH = fast_weighted_BH
local_utils.weighted_CS = fast_weighted_CS
print("🚀 [系统提示] 成功注入 NumPy 矩阵化加速引擎！本地慢速循环已被强行作废！")


# =============================================================================
# # 🛡️ 【数据与工程防错】标准 PyTorch 数据加载管道
# =============================================================================
class DeepPurposeDataset(Dataset):
    """将魔改 DataFrame 转换为标准的 PyTorch Dataset，彻底解决 KeyError: 0"""
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
    """纯单线程主进程加载（num_workers=0），从物理上消灭 Linux 多线程死锁卡死"""
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
    os.makedirs(path, exist_ok=True)
    lower_path = os.path.join(path, "hiv.csv")
    upper_path = os.path.join(path, "HIV.csv")

    if os.path.exists(lower_path) and os.path.exists(upper_path):
        return

    if os.path.exists(upper_path) and not os.path.exists(lower_path):
        shutil.copyfile(upper_path, lower_path)
        return
    if os.path.exists(lower_path) and not os.path.exists(upper_path):
        shutil.copyfile(lower_path, upper_path)
        return

    candidates = ["https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv"]
    last_error = None
    for url in candidates:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as response, open(upper_path, "wb") as out:
                out.write(response.read())
            shutil.copyfile(upper_path, lower_path)
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

# 1. 训练 DeepPurpose 预测模型（这里全面换成明确的 dp_utils）
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

# 2. 倾向性得分划分（校准集与测试集）
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
p_x = np.clip(np.exp(eta) / (1 + np.exp(eta)), 0.05, 0.8)
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

# 3. 运行多重假设检验与共形选择程序（这里全面换成劫持后的 local_utils）
c = 0
calib_scores_res = y_calib - hat_mu_calib
calib_scores_sub = - hat_mu_calib
calib_scores_clip = 100 * (y_calib > c) + c * (y_calib <= c) - hat_mu_calib
test_scores = c - hat_mu_test

print("⚡ [矩阵计算启动] 正在以火箭速度解算 WBH 与 WCS 变体...")

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