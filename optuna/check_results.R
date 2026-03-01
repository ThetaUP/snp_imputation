library(dplyr)
library(readr)
library(purrr)

base_dir <- "optuna/optuna_runs"
PERCENT = 60

# helper to read and round numeric columns
read_and_round <- function(file_path) {
  read_csv(file_path, show_col_types = FALSE) %>%
    mutate(across(where(is.numeric), ~ round(.x, 3)))
}

# list and combine training metrics
train_files <- list.files(base_dir, pattern = "^train_metrics\\.csv$", recursive = TRUE, full.names = TRUE)

train_data <- map_dfr(train_files, read_and_round)

# list and combine test metrics
test_files <- list.files(base_dir, pattern = "^snp_eval_metrics\\.csv$", recursive = TRUE, full.names = TRUE)

test_data <- map_dfr(test_files, read_and_round) %>%
  filter(model_path %in% train_data$model_path) %>%
  filter(grepl(paste0(PERCENT, ".0pct"), dataset)) %>%
  dplyr::distinct(model, dataset, f1_micro, f1_macro, f1_0, f1_1, f1_2, n_params, .keep_all = TRUE)

# merge train and test data and sort by val_f1_macro
all_data <- inner_join(train_data %>% filter(model_path %in% test_data$model_path),
                       test_data, by = c("model", "model_path")) %>%
  arrange(desc(val_f1_macro))

# view results
View(all_data)
