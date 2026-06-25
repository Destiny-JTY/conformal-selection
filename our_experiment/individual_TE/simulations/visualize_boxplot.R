###############################################################################
# 实验一箱线图: regression score, FDP & Power 分布 (20 seed)
###############################################################################

library(ggplot2)
library(dplyr)

# 读取所有单次运行结果
files <- list.files("./results_semi_synthetic/",
                    pattern = "coupling.*_seed.*\\.csv$", full.names = TRUE)
df <- do.call(rbind, lapply(files, read.csv))

# 只保留 regression, 去掉 dtm 和 UCS
keep_methods <- c("uBH", "wBH", "wCS_hete", "wCS_homo")
df <- df[df$score == "regression" & df$method %in% keep_methods, ]

# 标签
df$label <- df$method
df$label <- gsub("_", ".", df$label)
method_order <- c("uBH", "wBH", "wCS_hete", "wCS_homo")
label_order <- c("BH", "wBH", "CS.hete", "CS.homo")

# coupling 名称
df$coupling <- factor(df$coupling,
                       levels = c("independent", "positive", "negative"),
                       labels = c("Independent", "Positive", "Negative"))

# ============================================================================
# 图1: FDP 箱线图
# ============================================================================
p1 <- df %>%
  ggplot(aes(x = factor(method, levels = method_order, labels = label_order),
             y = fdp, fill = coupling)) +
  geom_boxplot(outlier.size = 0.8, alpha = 0.85) +
  geom_hline(yintercept = 0.1, linetype = "dashed", color = "grey40", size = 0.5) +
  geom_hline(yintercept = 0.2, linetype = "dashed", color = "grey40", size = 0.5) +
  facet_wrap(~ q, labeller = labeller(q = function(x) paste0("FDR = ", x))) +
  scale_fill_brewer(palette = "Set1") +
  labs(title = "FDP (regression score, 20 seeds)",
       y = "FDP", x = "", fill = "Coupling") +
  theme_minimal()

ggsave("./results_semi_synthetic/fig_boxplot_fdp.png", p1,
       width = 12, height = 5, dpi = 150)
cat("FDP boxplot saved\n")

# ============================================================================
# 图2: Power 箱线图
# ============================================================================
p2 <- df %>%
  ggplot(aes(x = factor(method, levels = method_order, labels = label_order),
             y = power, fill = coupling)) +
  geom_boxplot(outlier.size = 0.8, alpha = 0.85) +
  facet_wrap(~ q, labeller = labeller(q = function(x) paste0("FDR = ", x))) +
  scale_fill_brewer(palette = "Set1") +
  labs(title = "Power (regression score, 20 seeds)",
       y = "Power", x = "", fill = "Coupling") +
  theme_minimal()

ggsave("./results_semi_synthetic/fig_boxplot_power.png", p2,
       width = 12, height = 5, dpi = 150)
cat("Power boxplot saved\n")
