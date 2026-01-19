# ==============================================================================
# CREDIT RISK SCORECARD - FINAL PROJECT SCRIPT
# Dataset: mortgage_sample.csv
# ==============================================================================

# 0. SETUP & LIBRARIES
# ------------------------------------------------------------------------------
if(!require(scorecard)) install.packages("scorecard")
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(caret)) install.packages("caret")
if(!require(pROC)) install.packages("pROC")

library(scorecard)  # Binning, WoE, PSI, Scaling
library(tidyverse)  # Data manipulation
library(caret)      # Correlation functions
library(pROC)       # AUC/ROC calculations

# 1. DATA LOAD & PREPARATION
# ------------------------------------------------------------------------------
# [SLIDE 2: DATA SCOPE]
data_raw <- read.csv("mortgage_sample.csv")

# Filter for Public sample only (Good Practice)
df_public <- data_raw %>% filter(sample == "public")

# Define Target and ID variables to exclude from predictors
target_var <- "default_time"
vars_exclude <- c("id", "time", "orig_time", "first_time", "mat_time",
                  "payoff_time", "status_time", "sample", "default_time", "TARGET")

# Initial list of inputs (Predictors)
inputs <- setdiff(names(df_public), vars_exclude)

# [SLIDE 2: VALIDATION STRATEGY]
# Time-Based Split (70% Train / 30% Test)
# We split by 'time' variable to simulate Out-of-Time (OOT) validation
time_cutoff <- quantile(df_public$time, 0.7)
dt_train <- df_public %>% filter(time <= time_cutoff)
dt_test  <- df_public %>% filter(time > time_cutoff)

cat("Training Data (Time 0-", time_cutoff, "): ", nrow(dt_train), " rows\n", sep="")
cat("Testing Data (Time ", time_cutoff, "+): ", nrow(dt_test), " rows\n", sep="")

# 2. MISSING VALUES ANALYSIS
# ------------------------------------------------------------------------------
# [SLIDE 3: DATA CLEANING]
# Calculate missing percentage
missing_rates <- sapply(df_public[inputs], function(x) sum(is.na(x)) / length(x))
vars_too_missing <- names(missing_rates[missing_rates > 0.5])

if(length(vars_too_missing) > 0) {
    cat("Dropping variables with >50% missing:", paste(vars_too_missing, collapse=", "), "\n")
    inputs <- setdiff(inputs, vars_too_missing)
}

# 3. WOE BINNING (FEATURE ENGINEERING)
# ------------------------------------------------------------------------------
# [SLIDE 3 & 4: TRANSFORMATION]
# Generate Bins (Handles outliers and missing values automatically)
bins <- woebin(dt_train, y = target_var, x = inputs, min_perc_total = 0.05)

# ** FOR SLIDE 4 VISUAL **
# Plot the binning for a key variable (e.g., FICO or LTV) to show bad rate curve
woebin_plot(bins$FICO_orig_time)
# (Export this plot for your presentation)

# 4. VARIABLE SELECTION (IV & CORRELATION)
# ------------------------------------------------------------------------------
# [SLIDE 5: SELECTION FUNNEL]

# A. Information Value (IV) Filter > 0.02
iv_values <- iv(dt_train, y = target_var, x = inputs)
strong_vars <- iv_values %>% filter(info_value > 0.02) %>% pull(variable)
cat("Variables kept after IV filter (>0.02):", length(strong_vars), "\n")

# Apply WoE values for correlation check
train_woe <- woebin_ply(dt_train, bins, columns = strong_vars)
test_woe  <- woebin_ply(dt_test, bins, columns = strong_vars)

# B. Spearman Correlation Filter > 0.5 (Strict)
cor_mat <- cor(train_woe %>% select(all_of(strong_vars)), method = "spearman")
high_corr_idx <- findCorrelation(cor_mat, cutoff = 0.5)
high_corr_vars <- strong_vars[high_corr_idx]

# Final Predictor List
final_vars <- setdiff(strong_vars, high_corr_vars)
cat("Final Shortlist after Correlation filter:", length(final_vars), "\n")
print(final_vars)

# 5. MODELLING (LOGISTIC REGRESSION)
# ------------------------------------------------------------------------------
# [SLIDE 4: THE MODEL]
train_final <- train_woe %>% select(all_of(c(target_var, final_vars)))
test_final  <- test_woe  %>% select(all_of(c(target_var, final_vars)))

# Stepwise Selection
m_full <- glm(as.formula(paste(target_var, "~ .")), family = binomial(), data = train_final)
m_step <- step(m_full, direction = "both", trace = 0)

# View Coefficients (Log Odds)
summary(m_step)

# 6. SCORECARD SCALING
# ------------------------------------------------------------------------------
# [SLIDE 4 & 8: SCORECARD POINTS]
# Base Score=600, Odds=50:1, PDO=20
card <- scorecard(bins, m_step, points0 = 600, odds0 = 1/50, pdo = 20)

# Calculate Scores
train_score <- scorecard_ply(dt_train, card)
test_score  <- scorecard_ply(dt_test, card)

# ** FOR SLIDE 8 **
# View the points for the first variable to put in your table
print(card[[1]])

# 7. PERFORMANCE & VALIDATION
# ------------------------------------------------------------------------------
# [SLIDE 5: DISCRIMINATION / ROC]
pred_train <- predict(m_step, newdata = train_final, type = "response")
pred_test  <- predict(m_step, newdata = test_final, type = "response")

# ROC & GINI
roc_train <- roc(dt_train[[target_var]], pred_train)
roc_test  <- roc(dt_test[[target_var]], pred_test)

auc_train <- auc(roc_train)
auc_test  <- auc(roc_test)

# ** METRICS FOR SLIDE 5 **
cat(sprintf("Train GINI: %.2f%%\n", (2*auc_train-1)*100))
cat(sprintf("Test GINI:  %.2f%%\n", (2*auc_test-1)*100))

# ** PLOT FOR SLIDE 5 **
plot(roc_train, col="blue", main="ROC Curve: Train vs Test")
lines(roc_test, col="red")
legend("bottomright", legend=c("Train (Time 0-70%)", "Test (Time 70%+)"),
       col=c("blue", "red"), lwd=2)

# [SLIDE 6: STABILITY / PSI]
# ** PLOT FOR SLIDE 6 **
psi_out <- perf_psi(
    score = list(train = train_score, test = test_score),
    label = list(train = dt_train[[target_var]], test = dt_test[[target_var]])
)
psi_out$pic  # This generates the PSI bar charts

# [SLIDE 7: CALIBRATION]
# ** PLOT FOR SLIDE 7 **
perf_eva(pred = pred_train, label = dt_train[[target_var]], title = "Calibration")
