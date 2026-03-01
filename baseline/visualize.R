library(data.table)
library(stringr)
library(ggpubr)
library(tidyr)
library(dplyr)

# read data
df <- fread("baseline/beagle/evaluation.csv") %>%
  filter(grepl("2026-01-09", date))
df$percent <- as.numeric(str_extract(df$dataset, "(?<=_)\\d+"))

# plot per class f1
df_long <- df %>%
  arrange(percent) %>%
  select(percent, f1_0, f1_1, f1_2) %>%
  pivot_longer(cols = starts_with("f1_"),
               names_to = "class",
               values_to = "f1_score") %>%
  mutate(class = recode(class, f1_0="0", f1_1="1", f1_2="2"),
         f1_score = round(f1_score,2))

p <- ggbarplot(df_long, x="percent", y="f1_score",
          fill="class", color="gray", 
          position=position_dodge2(0.7, preserve = "single"),
          label=T) +
  scale_fill_brewer(palette="Set2") +
  scale_y_continuous(limits=c(0,1),breaks=seq(0, 1, 0.1)) +
  scale_x_discrete(labels = paste0(unique(df_long$percent), "%")) +
  labs(x="Missing fraction (%)", y="Per-genotype F1 score", fill="Genotype")

p
ggsave("baseline/beagle/evaluation.png", plot=p, width=13.5, height=5, dpi=300)
