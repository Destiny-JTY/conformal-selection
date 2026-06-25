###############################################################################
# 实验一批处理: 半合成 × 20 seed × 3 coupling
# 用法: Rscript run_batch.R
# 输出: results_semi_synthetic/ 下每个 seed 一份 CSV,
#       最后汇总为 summary_semi_synthetic.csv
###############################################################################

seeds <- 1:20
couplings <- 1:3       # 1=独立, 2=正相关, 3=负相关

for (coupling in couplings) {
  for (seed in seeds) {
    cmd <- sprintf("Rscript semi_synthetic.R %d %d", coupling, seed)
    cat(sprintf("\n>>> [coupling=%d, seed=%d] %s\n", coupling, seed, cmd))
    system(cmd)
  }
}

# ============================================================================
# 汇总所有结果
# ============================================================================
cat("\n>>> 汇总结果...\n")

all_files <- list.files("./results_semi_synthetic/",
                        pattern = "coupling.*_seed.*\\.csv$",
                        full.names = TRUE)

if (length(all_files) == 0) {
  stop("未找到结果文件")
}

all_data <- do.call(rbind, lapply(all_files, read.csv))
cat(sprintf("  共加载 %d 次运行, %d 条记录\n",
            length(unique(paste(all_data$coupling, all_data$seed))),
            nrow(all_data)))

# 按 coupling × score × method × weighted × q 汇总
library(dplyr)

summary <- all_data %>%
  group_by(coupling, score, method, weighted, q) %>%
  summarise(
    n_runs    = n(),
    fdp_mean  = mean(fdp, na.rm = TRUE),
    fdp_sd    = sd(fdp, na.rm = TRUE),
    power_mean = mean(power, na.rm = TRUE),
    power_sd  = sd(power, na.rm = TRUE),
    nrej_mean = mean(nrej, na.rm = TRUE),
    nrej_sd   = sd(nrej, na.rm = TRUE),
    .groups    = "drop"
  )

write.csv(as.data.frame(summary), "./results_semi_synthetic/summary.csv",
          row.names = FALSE)

# 打印关键对比
cat("\n========== 汇总: 加权 vs 不加权 FDP (按 coupling × score × q) ==========\n")
cat(sprintf("%-12s %-5s %-12s %-12s %8s %8s %8s\n",
            "Coupling", "q", "Score", "Method", "FDP_w", "FDP_u", "Delta"))
cat(strrep("-", 75), "\n")

for (cpl in c("independent", "positive", "negative")) {
  for (qq in unique(summary$q)) {
    for (sc in unique(summary$score)) {
      for (bm in c("BH", "CS.hete", "CS.homo", "CS.dtm")) {
        wn <- paste0("w", bm); un <- paste0("u", bm)
        rw <- summary[summary$coupling == cpl & summary$q == qq &
                      summary$score == sc & summary$method == wn, ]
        ru <- summary[summary$coupling == cpl & summary$q == qq &
                      summary$score == sc & summary$method == un, ]
        if (nrow(rw) > 0 && nrow(ru) > 0) {
          cat(sprintf("%-12s %-5.1f %-12s %-12s %8.4f %8.4f %+8.4f\n",
                      cpl, qq, sc, bm,
                      rw$fdp_mean, ru$fdp_mean, ru$fdp_mean - rw$fdp_mean))
        }
      }
    }
  }
}

cat("\n汇总已保存: ./results_semi_synthetic/summary.csv\n")
