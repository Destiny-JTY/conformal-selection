###############################################################################
# 实验一可视化: 只保留 uBH / wBH / wCS.hete / wCS.homo / wCS.dtm
###############################################################################

library(ggplot2)
library(dplyr)
library(tidyr)

df <- read.csv("./results_semi_synthetic/summary.csv")

# 只保留需要的 method
keep_methods <- c("uBH", "wBH", "wCS_hete", "wCS_homo", "wCS_dtm")
df <- df[df$method %in% keep_methods, ]

# 聚合 score: coupling + method + q
agg <- df %>%
  group_by(coupling, method, q) %>%
  summarise(fdp   = mean(fdp_mean),
            power = mean(power_mean),
            nrej  = mean(nrej_mean),
            .groups = "drop")

# 统一横轴标签
agg$label <- agg$method
agg$label <- gsub("_", ".", agg$label)

# 横轴顺序
x_order <- c("uBH", "wBH", "wCS_hete", "wCS_homo", "wCS_dtm")
x_labels <- c("BH", "wBH", "CS.hete", "CS.homo", "CS.dtm")

# ============================================================================
# 图1: FDP 柱状图
# ============================================================================
p1 <- agg %>%
  ggplot(aes(x = factor(method, levels = x_order, labels = x_labels),
             y = fdp, fill = coupling)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.6) +
  geom_text(aes(label = sprintf("%.4f", fdp)),
            position = position_dodge(width = 0.8),
            vjust = -0.5, size = 2.8) +
  geom_hline(yintercept = 0.1, linetype = "dashed", color = "grey50", size = 0.4) +
  geom_hline(yintercept = 0.2, linetype = "dashed", color = "grey50", size = 0.4) +
  facet_wrap(~ q, labeller = labeller(q = function(x) paste0("FDR = ", x))) +
  scale_fill_brewer(palette = "Set1", name = "Coupling") +
  labs(title = "FDP: all methods well below nominal q",
       y = "FDP", x = "") +
  theme_minimal()

ggsave("./results_semi_synthetic/fig1_fdp_bar.png", p1,
       width = 12, height = 5, dpi = 150)
cat("Fig1 saved\n")

# ============================================================================
# 图2: FDP Heatmap (coupling x method, 分 q 和 score)
# ============================================================================
agg_score <- df %>%
  group_by(coupling, score, method, q) %>%
  summarise(fdp = mean(fdp_mean), .groups = "drop")
agg_score$label <- gsub("_", ".", agg_score$method)

p2 <- agg_score %>%
  ggplot(aes(x = factor(method, levels = x_order, labels = x_labels),
             y = score, fill = fdp)) +
  geom_tile(color = "white", size = 0.5) +
  geom_text(aes(label = sprintf("%.4f", fdp)), size = 3) +
  facet_grid(coupling ~ q) +
  scale_fill_gradient(low = "white", high = "#d62728",
                       name = "FDP", limits = c(0, NA)) +
  labs(title = "FDP Heatmap: coupling x score x method",
       x = "", y = "Score") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 8))

ggsave("./results_semi_synthetic/fig2_fdp_heatmap.png", p2,
       width = 13, height = 7, dpi = 150)
cat("Fig2 saved\n")

# ============================================================================
# 图3: Power 柱状图
# ============================================================================
p3 <- agg %>%
  ggplot(aes(x = factor(method, levels = x_order, labels = x_labels),
             y = power, fill = coupling)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.6) +
  geom_text(aes(label = sprintf("%.3f", power)),
            position = position_dodge(width = 0.8),
            vjust = -0.5, size = 2.8) +
  facet_wrap(~ q, labeller = labeller(q = function(x) paste0("FDR = ", x))) +
  scale_fill_brewer(palette = "Set1", name = "Coupling") +
  labs(title = "Power: negative coupling > independent > positive",
       y = "Power", x = "") +
  theme_minimal()

ggsave("./results_semi_synthetic/fig3_power_bar.png", p3,
       width = 12, height = 5, dpi = 150)
cat("Fig3 saved\n")

# ============================================================================
# 汇总表
# ============================================================================
cat("\n========== Summary (across scores) ==========\n")
cat(sprintf("%-12s %-12s %5s %8s %8s %8s\n",
            "Coupling", "Method", "q", "FDP", "Power", "Nsel"))
cat(strrep("-", 55), "\n")
for (cpl in c("independent", "positive", "negative")) {
  sub <- agg[agg$coupling == cpl, ]
  for (m in x_order) {
    for (qv in c(0.1, 0.2)) {
      r <- sub[sub$method == m & sub$q == qv, ]
      if (nrow(r) > 0) {
        cat(sprintf("%-12s %-12s %5.1f %8.4f %8.4f %8.1f\n",
                    cpl, m, qv, r$fdp, r$power, r$nrej))
      }
    }
  }
}
