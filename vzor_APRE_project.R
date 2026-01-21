# ==============================================================================
# PROFESSIONAL CREDIT RISK SCORECARD (FINAL RANDOM SPLIT VERSION)
# Project: US Residential Mortgage PD Model
# Fixes: Overfitting (GINI Gap) & PSI NULL Error & Syntax Errors
# ==============================================================================

# 0. ENVIRONMENT SETUP
# ------------------------------------------------------------------------------
if(!require(scorecard)) install.packages("scorecard")
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(caret))     install.packages("caret")
if(!require(pROC))      install.packages("pROC")
if(!require(car))       install.packages("car")
if(!require(zoo))       install.packages("zoo")

library(scorecard)
library(tidyverse)
library(caret)
library(pROC)
library(car)
library(zoo)

# 1. DATA IMPORT & FLEXIBLE COHORT (SMART TARGET LOGIC)
# ------------------------------------------------------------------------------
data_raw <- read.csv("mortgage_sample.csv")

# A. Split Public/Private
df_public_raw  <- data_raw %>% filter(sample == "public")
df_private_raw <- data_raw %>% filter(sample == "private")

# B. Custom Target Function (Smart Logic)
# Keeps "Short Defaults", Drops "Short Goods"
calc_target_12m <- function(x) {
    if(any(x == 1, na.rm = TRUE)) {
        return(1)
    } else if(length(x) < 12) {
        return(NA)
    } else {
        return(0)
    }
}

cat("Step 1: Engineering Target (Smart Logic)...\n")

# C. Apply Logic to Public Data
df_cohort <- df_public_raw %>%
    arrange(id, time) %>%
    group_by(id) %>%
    mutate(
        target_12m = rollapply(default_time, width = 12, FUN = calc_target_12m,
                               align = "left", fill = NA, partial = TRUE),
        mob = time - first_time
    ) %>%
    # Flexible Cohort: Keep first observation + every 12 months
    filter((mob %% 12) == 0) %>%
    filter(!is.na(target_12m)) %>%
    ungroup() %>%
    select(-mob)

# D. Private Sample (for Deployment check)
df_private <- df_private_raw

cat("Final Analysis Rows:    ", nrow(df_cohort), "\n")
cat("Bad Rate (12m Window):  ", sprintf("%.2f%%", mean(df_cohort$target_12m)*100), "\n")


# 2. VARIABLE SELECTION
# ------------------------------------------------------------------------------
target_var <- "target_12m"
vars_exclude <- c("id", "time", "orig_time", "first_time", "mat_time",
                  "payoff_time", "status_time", "sample", "default_time")

inputs <- setdiff(names(df_cohort), c(vars_exclude, target_var))

# Check for Missing Values (>50% Removal)
missing_rates <- sapply(df_cohort[inputs], function(x) sum(is.na(x)) / length(x))
vars_too_missing <- names(missing_rates[missing_rates > 0.5])
if(length(vars_too_missing) > 0) inputs <- setdiff(inputs, vars_too_missing)


# 3. SPLITTING & BINNING (THE FIX: RANDOM SPLIT)
# ------------------------------------------------------------------------------
# We switch to Random Split to cure the "Time-Split" Overfitting
set.seed(123)
train_index <- createDataPartition(df_cohort[[target_var]], p = 0.7, list = FALSE)

dt_train <- df_cohort[train_index, ]
dt_test  <- df_cohort[-train_index, ]

cat("Step 3: Random Split Complete (70/30).\n")
cat("Train Rows:", nrow(dt_train), " | Test Rows:", nrow(dt_test), "\n")

# Run WoE Binning
cat("Running Weight of Evidence Binning...\n")
bins <- woebin(dt_train, y = target_var, x = inputs, min_perc_total = 0.05)


# 4. FEATURE SELECTION (IV & CORRELATION)
# ------------------------------------------------------------------------------
# A. IV Selection (> 0.02)
iv_values <- iv(dt_train, y = target_var, x = inputs)
strong_vars <- iv_values %>% filter(info_value > 0.02) %>% pull(variable)

# Apply WoE Transformation
train_woe <- woebin_ply(dt_train, bins)
test_woe  <- woebin_ply(dt_test, bins)

# B. Correlation Check (Fixing the _woe suffix issue)
strong_vars_woe <- paste0(strong_vars, "_woe")
cor_mat <- cor(train_woe %>% select(all_of(strong_vars_woe)), method = "spearman")
high_corr_idx <- findCorrelation(cor_mat, cutoff = 0.5)
high_corr_vars_woe <- strong_vars_woe[high_corr_idx]

# Final Variables for Model
final_vars_woe <- setdiff(strong_vars_woe, high_corr_vars_woe)
cat("\n[SELECTION] Final Variables kept:", length(final_vars_woe), "\n")


# 5. MODELLING (STEPWISE)
# ------------------------------------------------------------------------------
train_final <- train_woe %>% select(all_of(c(target_var, final_vars_woe)))
test_final  <- test_woe  %>% select(all_of(c(target_var, final_vars_woe)))

m_full <- glm(as.formula(paste(target_var, "~ .")), family = binomial(), data = train_final)
m_step <- step(m_full, direction = "both", trace = 0)

cat("\n--- MODEL COEFFICIENTS ---\n")
summary(m_step) # Check for Positive Estimates and *** Stars


# 6. SCORECARD SCALING & PSI
# ------------------------------------------------------------------------------
card <- scorecard(bins, m_step, points0 = 500, odds0 = 1/50, pdo = 50)

# Calculate Scores
train_score <- scorecard_ply(dt_train, card)
test_score  <- scorecard_ply(dt_test, card)
private_scores <- scorecard_ply(df_private, card) # For Deployment Check

# 7. PERFORMANCE & DEPLOYMENT CHECK (FIXED)
# ------------------------------------------------------------------------------
# A. GINI Gap Check
train_gini <- (2 * auc(roc(dt_train[[target_var]], train_score$score)) - 1) * 100
test_gini  <- (2 * auc(roc(dt_test[[target_var]], test_score$score)) - 1) * 100

cat(sprintf("\n--- PERFORMANCE ---\nTrain GINI: %.2f%%\nTest GINI:  %.2f%%\nGap:        %.2f points\n",
            train_gini, test_gini, train_gini - test_gini))

# B. Deployment PSI (Fixed: No Labels, Robust Vector Method)
cat("\n--- DEPLOYMENT PSI (STABILITY) ---\n")
psi_check <- perf_psi(score = list(train = train_score$score, private = private_scores$score))
print(psi_check$psi)

# C. VIF Check
cat("\n--- VIF CHECK ---\n")
try(print(car::vif(m_step)), silent=TRUE)

# 8. EXPORT
df_private$Score <- private_scores$score
write.csv(df_private %>% select(id, Score), "final_private_results.csv", row.names=FALSE)
cat("\nScript Complete. Results saved.\n")
