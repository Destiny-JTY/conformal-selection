###############################################################################
# 实验一: ITE 半合成实验
#
# 真实协变量 (ACIC X1-X5) + 参数化 outcome → ground truth 已知
#
# 关键设计:
#   - 倾向性得分用 pbeta (非 logit), 避免被 logit 回归直接猜中
#   - 分析阶段用 RF 估计 ê(x), 模拟真实场景中的模型误设
#   - 所有 5 个协变量均在 DGP 中有结构性作用
#   - 处理效应 τ 显式建模
#
# 用法:
#   Rscript semi_synthetic.R <coupling> <seed>
#   coupling: 1=独立, 2=正相关, 3=负相关
#   例: Rscript semi_synthetic.R 1 42
###############################################################################

library(tidyverse)
library(grf)

# ============================================================================
# 0. 加载工具函数 (w_BH, w_CS, eval.FDR 定义在 util_simu.R)
# ============================================================================
source("../util_simu.R")

args <- commandArgs(trailingOnly = TRUE)
coupling_type <- as.integer(args[1])
seed          <- as.integer(args[2])
set.seed(seed)

# ============================================================================
# 1. 参数
# ============================================================================
noise_sig <- 0.3
n_total   <- 5000
n_train   <- 2500
n_calib   <- 1250
n_test    <- 1250
q_levels  <- c(0.1, 0.2)

# ============================================================================
# 2. 加载真实协变量 & ecdf 映射到 [0,1]
# ============================================================================
cat(">>> 加载 ACIC 协变量 X1-X5, ecdf 映射到 U1-U5\n")
raw  <- read.csv("../realdata/acic_data.csv")
Xraw <- raw %>% select(X1, X2, X3, X4, X5) %>% as.matrix()
Xraw <- Xraw[1:n_total, ]

U <- matrix(0, n_total, 5)
for (j in 1:5) {
  U[, j] <- ecdf(Xraw[, j])(Xraw[, j]) + runif(n_total, 0, 1e-6)
  U[, j] <- pmin(pmax(U[, j], 1e-6), 1 - 1e-6)
}
colnames(U) <- paste0("U", 1:5)

# ============================================================================
# 3. 倾向性得分 (pbeta 形式——logit 回归无法精确恢复)
# ============================================================================
# e(U) = 0.15 + 0.30 * pbeta(U1, 2, 4) + 0.08 * U5
#
# pbeta(x,2,4): U1 低时增长快, U1 高时平缓 → logit 无法拟合的曲率
# U5 附加线性成分 → 多维协变量偏移
# 值域约 [0.15, 0.53], 保证 overlap

cat(">>> 生成 e(U) = 0.15 + 0.30*pbeta(U1,2,4) + 0.08*U5\n")
e_x <- 0.15 + 0.30 * pbeta(U[, 1], 2, 4) + 0.08 * U[, 5]
T_all <- rbinom(n_total, 1, e_x)

# ============================================================================
# 4. sigmoid 核
# ============================================================================
sigmoid <- function(z) 1 / (1 + exp(-12 * (z - 0.5)))
s1  <- sigmoid(U[, 1])
s2  <- sigmoid(U[, 2])
s12 <- s1 * s2

# ============================================================================
# 5. 基线函数 f(U) (5 个协变量全部参与)
# ============================================================================
# f(U) = 1.5 + 1.5*s(U1)*s(U2) + 0.4*U3 + 0.3*U4*U5
#   项                范围      含义
#   1.5               常数      基础水平
#   1.5*s12           0~1.5     U1,U2 都高时激活
#   0.4*U3            0~0.4     线性效果
#   0.3*U4*U5         0~0.3     U4,U5 协同

cat(">>> f(U) = 1.5 + 1.5*s(U1)*s(U2) + 0.4*U3 + 0.3*U4*U5\n")
f_U <- 1.5 + 1.5 * s12 + 0.4 * U[, 3] + 0.3 * U[, 4] * U[, 5]

# ============================================================================
# 6. 处理效应 τ(U) (U1,U2,U3,U4 参与)
# ============================================================================
# τ(U) = 2.5*s(U1)*s(U2) + 0.5*(U3-0.5) + 0.2*U4
#   项                   范围         含义
#   2.5*s12              0~2.5        核心效应异质性
#   0.5*(U3-0.5)         -0.25~0.25   U3 微调 (均值中心化)
#   0.2*U4               0~0.2        U4 附加效应
#   τ 范围约 [-0.25, 2.95]

cat(">>> τ(U) = 2.5*s(U1)*s(U2) + 0.5*(U3-0.5) + 0.2*U4\n")
tau_U <- 2.5 * s12 + 0.5 * (U[, 3] - 0.5) + 0.2 * U[, 4]

# ============================================================================
# 7. 潜在结果 Y(0), Y(1)
# ============================================================================
# Y(0) = f(U) + ε₀
# Y(1) = f(U) + τ(U) + ε₁
# coupling: 独立 (ε₀⊥ε₁) / 正相关 (ε₀=ε₁) / 负相关 (ε₀=-ε₁)

cat(">>> 生成 Y(0), Y(1)\n")
eta <- rnorm(n_total, 0, noise_sig)
eps <- rnorm(n_total, 0, noise_sig)

if (coupling_type == 1) {
  e0 <- eps; e1 <- eta; coupling_label <- "independent"
} else if (coupling_type == 2) {
  e0 <- eta; e1 <- eta; coupling_label <- "positive"
} else {
  e0 <- -eta; e1 <- eta; coupling_label <- "negative"
}
cat(sprintf("    coupling = %s\n", coupling_label))

Y0 <- f_U + e0
Y1 <- f_U + tau_U + e1
Y_obs <- T_all * Y1 + (1 - T_all) * Y0

# ============================================================================
# 8. 数据划分 (自适应: T=1 数量有限, 动态调整)
# ============================================================================
reind <- sample(n_total, n_total)

# 可用 T=1 和 T=0 的索引
idx_T1 <- which(T_all[reind] == 1)
idx_T0 <- which(T_all[reind] == 0)
n_T1 <- length(idx_T1)

# 自适应调整各集大小
n_train_use <- min(n_train, floor(n_T1 * 0.7))
n_calib_use <- min(n_calib, n_T1 - n_train_use)
n_test_use  <- min(n_test,  length(idx_T0))

cat(sprintf(">>> 划分: 训练=%d 校准=%d 测试=%d (可用T1=%d)\n",
            n_train_use, n_calib_use, n_test_use, n_T1))

# 训练集: T=1
t1_idx <- reind[idx_T1[1:n_train_use]]
X_train <- U[t1_idx, , drop = FALSE]
Y_train <- Y_obs[t1_idx]

# 校准集: T=1
calib_sub <- reind[idx_T1[(n_train_use + 1):(n_train_use + n_calib_use)]]
X_calib   <- U[calib_sub, , drop = FALSE]
Y_calib   <- Y_obs[calib_sub]
T_calib   <- T_all[calib_sub]
ncal      <- length(Y_calib)

# 测试集: T=0 (反事实)
test_sub <- reind[idx_T0[1:n_test_use]]
X_test   <- U[test_sub, , drop = FALSE]
Yobs_test <- Y_obs[test_sub]
Y1_test   <- Y1[test_sub]
T_test    <- T_all[test_sub]
ex_test_true <- e_x[test_sub]  # 真实 e(x) 仅用于参考
ntest     <- length(test_sub)

# 选择目标: 治疗有效的个体 (Y(1) > Y(0))
# 注意: 测试集 T=0, Yobs_test = Y(0), Y1_test 是生成的真值
truths <- Y1_test <= Yobs_test    # TRUE = 治疗无效 = 假发现
n_pos  <- sum(!truths)
cat(sprintf("    测试集 n=%d, 真阳性(Y1>Y0)=%d (%.1f%%)\n",
            ntest, n_pos, 100*n_pos/ntest))

# ============================================================================
# 9. 估计倾向性得分 (RF——真实 e(x) 是 pbeta 形式, RF 只能近似)
# ============================================================================
cat(">>> 用 RF 估计 ê(x) (真实 e(x) 是 pbeta, logit 会误设)\n")

# 用全量数据估计 (X,T)
ps_rf <- regression_forest(U, T_all, num.threads = 1)
e_hat_calib <- predict(ps_rf, X_calib)$predictions
e_hat_test  <- predict(ps_rf, X_test)$predictions

# 截断避免极端权重
e_hat_calib <- pmin(pmax(e_hat_calib, 0.05), 0.95)
e_hat_test  <- pmin(pmax(e_hat_test,  0.05), 0.95)

# ============================================================================
# 10. 权重
# ============================================================================
pp <- mean(T_all)

# 加权 (基于估计的 ê(x))
cal.weight  <- ((1-pp) * e_hat_calib / (pp * (1-e_hat_calib)))^(-1)
test.weight <- ((1-pp) * e_hat_test  / (pp * (1-e_hat_test)))^(-1)

# 不加权
cal.weight.unif  <- rep(1, ncal)
test.weight.unif <- rep(1, ntest)

# ============================================================================
# 11. 训练预测模型 & 得分函数
# ============================================================================
cat(">>> 训练预测模型 (随机森林)\n")

# 回归森林 → 回归残差
reg_rf <- regression_forest(X_train, Y_train, num.threads = 1)
cal_score_reg  <- Y_calib - predict(reg_rf, X_calib)$predictions
test_score_reg <- Yobs_test - predict(reg_rf, X_test)$predictions

# 分位数森林 → CQR
qr_rf <- quantile_forest(X_train, Y_train, num.threads = 1)

pred_cal_02 <- predict(qr_rf, X_calib, quantiles = c(0.2))$predictions
pred_tst_02 <- predict(qr_rf, X_test,  quantiles = c(0.2))$predictions
cal_score_cqr02  <- Y_calib - as.numeric(pred_cal_02)
test_score_cqr02 <- Yobs_test - as.numeric(pred_tst_02)

pred_cal_05 <- predict(qr_rf, X_calib, quantiles = c(0.5))$predictions
pred_tst_05 <- predict(qr_rf, X_test,  quantiles = c(0.5))$predictions
cal_score_cqr05  <- Y_calib - as.numeric(pred_cal_05)
test_score_cqr05 <- Yobs_test - as.numeric(pred_tst_05)

# ============================================================================
# 12. 运行所有方法
# ============================================================================
cat(">>> 运行 8 方法 × 3 得分 × 2 FDR\n\n")

run_one <- function(cal_sc, test_sc, truths, sname) {
  rows <- list()
  for (q in q_levels) {
    Rw <- list(
      wBH      = w_BH(cal_sc, test_sc, cal.weight,  test.weight,  q),
      wCS_hete = w_CS(cal_sc, test_sc, cal.weight,  test.weight,  q, rand = 'hete'),
      wCS_homo = w_CS(cal_sc, test_sc, cal.weight,  test.weight,  q, rand = 'homo'),
      wCS_dtm  = w_CS(cal_sc, test_sc, cal.weight,  test.weight,  q, rand = 'dtm')
    )
    Ru <- list(
      uBH      = w_BH(cal_sc, test_sc, cal.weight.unif, test.weight.unif, q),
      uCS_hete = w_CS(cal_sc, test_sc, cal.weight.unif, test.weight.unif, q, rand = 'hete'),
      uCS_homo = w_CS(cal_sc, test_sc, cal.weight.unif, test.weight.unif, q, rand = 'homo'),
      uCS_dtm  = w_CS(cal_sc, test_sc, cal.weight.unif, test.weight.unif, q, rand = 'dtm')
    )
    for (m in names(Rw)) {
      ev <- eval.FDR(Rw[[m]], truths)
      rows[[length(rows)+1]] <- data.frame(
        score = sname, q = q, method = m, weighted = "Yes",
        nrej = ev$nrej, fdp = ev$fdp, power = ev$power,
        stringsAsFactors = FALSE
      )
    }
    for (m in names(Ru)) {
      ev <- eval.FDR(Ru[[m]], truths)
      rows[[length(rows)+1]] <- data.frame(
        score = sname, q = q, method = m, weighted = "No",
        nrej = ev$nrej, fdp = ev$fdp, power = ev$power,
        stringsAsFactors = FALSE
      )
    }
  }
  do.call(rbind, rows)
}

res_reg  <- run_one(cal_score_reg,  test_score_reg,  truths, "regression")
res_cqr2 <- run_one(cal_score_cqr02, test_score_cqr02, truths, "cqr_0.2")
res_cqr5 <- run_one(cal_score_cqr05, test_score_cqr05, truths, "cqr_0.5")

all_res <- rbind(res_reg, res_cqr2, res_cqr5)
all_res$coupling <- coupling_label
all_res$seed     <- seed

# ============================================================================
# 13. 输出
# ============================================================================
cat("======================================================================\n")
cat("  半合成实验\n")
cat("======================================================================\n")
cat("  e(U) = 0.15 + 0.30*pbeta(U1,2,4) + 0.08*U5   (pbeta→logit误设)\n")
cat("  f(U) = 1.5 + 1.5*s(U1)*s(U2) + 0.4*U3 + 0.3*U4*U5\n")
cat("  τ(U) = 2.5*s(U1)*s(U2) + 0.5*(U3-0.5) + 0.2*U4\n")
cat("  Y(0) = f(U) + ε0,  Y(1) = f(U) + τ(U) + ε1\n")
cat(sprintf("  选择目标: 选出 Y(1) > Y(0) 的个体 (治疗有效)\n"))
cat(sprintf("  coupling=%s, σ=%.1f, seed=%d\n",
            coupling_label, noise_sig, seed))
cat("======================================================================\n\n")

for (q in q_levels) {
  cat(sprintf("──── FDR=%.1f ────\n", q))
  sub <- all_res[all_res$q == q, ]
  cat(sprintf("%-12s %-8s %-12s %8s %8s %8s\n",
              "Method","Weighted","Score","FDP","Power","Nsel"))
  cat(strrep("-",62), "\n")
  for (i in 1:nrow(sub))
    cat(sprintf("%-12s %-8s %-12s %8.4f %8.4f %8d\n",
                sub$method[i], sub$weighted[i], sub$score[i],
                sub$fdp[i], sub$power[i], sub$nrej[i]))
  cat("\n")
}

cat("──── 加权 vs 不加权 FDP 差异 (预期 Δ>0, 不加权失控) ────\n")
cat(sprintf("%-12s %-12s %6s %8s %8s %8s\n",
            "Score","Method","q","FDP_w","FDP_u","Δ"))
cat(strrep("-",60), "\n")
for (sc in unique(all_res$score)) {
  for (q in q_levels) {
    sub <- all_res[all_res$score == sc & all_res$q == q, ]
    for (bm in c("BH","CS.hete","CS.homo","CS.dtm")) {
      wn <- paste0("w", bm); un <- paste0("u", bm)
      fw <- sub$fdp[sub$method == wn]; fu <- sub$fdp[sub$method == un]
      if (length(fw) && length(fu))
        cat(sprintf("%-12s %-12s %6.1f %8.4f %8.4f %+8.4f\n",
                    sc, bm, q, fw, fu, fu - fw))
    }
  }
}

# 保存
dir.create("./results_semi_synthetic/", showWarnings = FALSE, recursive = TRUE)
fname <- paste0("./results_semi_synthetic/coupling", coupling_type, "_seed", seed, ".csv")
write.csv(all_res, fname, row.names = FALSE)
cat(sprintf("\n已保存: %s\n", fname))
