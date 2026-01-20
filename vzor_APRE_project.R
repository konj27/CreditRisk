# ==============================================================================
# PROFESSIONAL CREDIT RISK SCORECARD (FINAL PRODUCTION SCRIPT)
# Project: US Residential Mortgage PD Model
# Strategy: Flexible Cohort (PIT) | 12-Month Performance Window | Macro Enabled
# ==============================================================================

# 0. ENVIRONMENT SETUP
# ------------------------------------------------------------------------------
# Install/Load necessary packages
if(!require(scorecard)) install.packages("scorecard")
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(caret))     install.packages("caret")
if(!require(pROC))      install.packages("pROC")
if(!require(car))       install.packages("car")  # For VIF Check
if(!require(zoo))       install.packages("zoo")  # For Target Engineering

library(scorecard)
library(tidyverse)
library(caret)
library(pROC)
library(car)
library(zoo)

# 1. DATA IMPORT & FLEXIBLE COHORT CONSTRUCTION [Ref: Slide 34]
# ------------------------------------------------------------------------------
# We use "Flexible Cohorts" to maximize data usage while maintaining independence.
# We use strict Target Engineering (partial=FALSE) to ensure verified outcomes.

data_raw <- read.csv("mortgage_sample.csv")

# A. Split Raw Data (Separate Private now to avoid leakage/errors)
df_public_raw  <- data_raw %>% filter(sample == "public")
df_private_raw <- data_raw %>% filter(sample == "private")

cat("Step 1: Engineering Target & Building Flexible Cohorts...\n")

# B. Engineer Target & Create Flexible Cohort (Public Only)
df_cohort <- df_public_raw %>%
    arrange(id, time) %>%
    group_by(id) %>%
    mutate(
        # 1. Engineer Target (Look forward 12 months) [Ref: Slide 34]
        # partial=FALSE: If we are at the end of data and can't see full 12m, return NA.
        # This is critical to avoid labeling struggling customers as "Good".
        target_12m = rollapply(default_time, width = 12, FUN = max,
                               align = "left", fill = NA, partial = FALSE),

        # 2. Identify "Start Time" for this specific customer
        start_time = min(time),

        # 3. Calculate "Months on Book" relative to THEIR start date
        mob = time - start_time
    ) %>%
    # 4. Filter for Flexible Cohort
    # Keep the First Observation (mob=0) and then every 12 months (mob=12, 24...)
    # This ensures Independence (Slide 34) regardless of calendar month.
    filter((mob %% 12) == 0) %>%
    filter(!is.na(target_12m)) %>% # Remove rows with incomplete future
    ungroup() %>%
    select(-start_time, -mob) # Clean up helper columns

# C. Private Sample (Keep full for deployment check)
df_private <- df_private_raw

cat("Original Public Rows:   ", nrow(df_public_raw), "\n")
cat("Final Flexible Cohort Rows: ", nrow(df_cohort), "\n")
cat("Bad Rate (12m Window):  ", sprintf("%.2f%%", mean(df_cohort$target_12m)*100), "\n")


# 2. VARIABLE SELECTION (MACRO-ENABLED) [Ref: Slide 48]
# ------------------------------------------------------------------------------
# We INCLUDE macro variables (gdp, uer, hpi) to test their predictive power.
target_var <- "target_12m"

# Exclude Identifiers, Dates, and "Leakage" (current status variables)
vars_exclude <- c("id", "time", "orig_time", "first_time", "mat_time",
                  "payoff_time", "status_time", "sample", "default_time")

# Create input list
inputs <- setdiff(names(df_cohort), c(vars_exclude, target_var))

# Check for Missing Values (>50% Removal)
missing_rates <- sapply(df_cohort[inputs], function(x) sum(is.na(x)) / length(x))
vars_too_missing <- names(missing_rates[missing_rates > 0.5])

if(length(vars_too_missing) > 0) {
    cat("\n[FILTER] Dropping variables with >50% missing:", paste(vars_too_missing, collapse=", "), "\n")
    inputs <- setdiff(inputs, vars_too_missing)
}

cat("\n[INFO] Variables included for modeling (includes Macro):", length(inputs), "\n")


# 3. SPLITTING & BINNING (WOE) [Ref: Slide 32]
# ------------------------------------------------------------------------------
# Time-Based Split (70% Train / 30% Test)
time_cutoff <- quantile(df_cohort$time, 0.7)
dt_train <- df_cohort %>% filter(time <= time_cutoff)
dt_test  <- df_cohort %>% filter(time > time_cutoff)

# WoE Binning (Handles Outliers & Missing Values automatically)
bins <- woebin(dt_train, y = target_var, x = inputs, min_perc_total = 0.05)

# PLOT: Visual Check for Macro Variable (Optional)
# If 'uer_time' exists, plot it to see risk separation
if("uer_time" %in% names(bins)) woebin_plot(bins$uer_time)


# 4. UNIVARIATE & MULTIVARIATE CHECKS [Ref: Slide 46]
# ------------------------------------------------------------------------------
# A. Univariate Selection (IV > 0.02)
iv_values <- iv(dt_train, y = target_var, x = inputs)
strong_vars <- iv_values %>% filter(info_value > 0.02) %>% pull(variable)

cat("\n[SELECTION] Variables kept (IV > 0.02):", length(strong_vars), "\n")
# Check if macro vars survived the IV filter
cat("Macro vars surviving IV:", paste(intersect(strong_vars, c("gdp_time", "uer_time", "hpi_time")), collapse=", "), "\n")

# B. Spearman Correlation Check (Threshold > 0.5)
# Note: Macro vars are often correlated. This step handles multicollinearity.
train_woe <- woebin_ply(dt_train, bins, columns = strong_vars)
test_woe  <- woebin_ply(dt_test, bins, columns = strong_vars)

cor_mat <- cor(train_woe %>% select(all_of(strong_vars)), method = "spearman")
high_corr_idx <- findCorrelation(cor_mat, cutoff = 0.5)
high_corr_vars <- strong_vars[high_corr_idx]
final_vars <- setdiff(strong_vars, high_corr_vars)

cat("[SELECTION] Dropped due to Correlation > 0.5:", paste(high_corr_vars, collapse=", "), "\n")

# C. VIF Check (Variance Inflation Factor)
cat("\n[CHECK] Calculating VIF...\n")
form_vif <- as.formula(paste(target_var, "~", paste(final_vars, collapse=" + ")))
m_vif_check <- glm(form_vif, data = train_woe, family = binomial())

vif_vals <- car::vif(m_vif_check)
print(vif_vals)

if(any(vif_vals > 5)) cat("WARNING: High VIF (>5) detected.\n") else cat("VIF Check Passed (All < 5).\n")


# 5. MODELLING (STEPWISE LOGISTIC REGRESSION) [Ref: Slide 40]
# ------------------------------------------------------------------------------
train_final <- train_woe %>% select(all_of(c(target_var, final_vars)))
test_final  <- test_woe  %>% select(all_of(c(target_var, final_vars)))

m_full <- glm(as.formula(paste(target_var, "~ .")), family = binomial(), data = train_final)
m_step <- step(m_full, direction = "both", trace = 0)

summary(m_step)
# Note: Look for '***' next to macro variables in the summary.


# 6. SCORECARD SCALING [Ref: Slide 50]
# ------------------------------------------------------------------------------
# Parameters: Base=500, Odds=1:50, PDO=50
# Note: Odds 1:50 corresponds to prob approx 1/51 as per Slide 50.
card <- scorecard(bins, m_step,
                  points0 = 500,
                  odds0   = 1/50,
                  pdo     = 50)

# Calculate Scores
train_score <- scorecard_ply(dt_train, card)
test_score  <- scorecard_ply(dt_test, card)

# Print Scorecard Points (For Slide 8)
print(card[[1]])


# 7. PERFORMANCE & DEPLOYMENT CHECK [Ref: Slide 68]
# ------------------------------------------------------------------------------
pred_train <- predict(m_step, newdata = train_final, type = "response")
pred_test  <- predict(m_step, newdata = test_final, type = "response")

# A. ROC & GINI
roc_train <- roc(dt_train[[target_var]], pred_train)
roc_test  <- roc(dt_test[[target_var]], pred_test)
gini_train <- (2 * auc(roc_train) - 1) * 100
gini_test  <- (2 * auc(roc_test) - 1) * 100

cat(sprintf("\nPERFORMANCE:\nTrain GINI: %.2f%%\nTest GINI:  %.2f%%\n", gini_train, gini_test))

# B. ROC Plot (For Presentation)
plot(roc_train, col="blue", main="ROC Curve: Train vs Test", lwd=2)
lines(roc_test, col="red", lwd=2)
legend("bottomright", legend=c("Train", "Test"), col=c("blue", "red"), lwd=2)

# C. Deployment PSI (Public vs Private)
# We apply the scorecard to the Private sample to check Stability.
private_scores <- scorecard_ply(df_private, card)

psi_deployment <- perf_psi(
    score = list(train = train_score, private = private_scores),
    label = list(train = dt_train[[target_var]], private = rep(0, nrow(df_private)))
)

cat("\nDeployment PSI (Public vs Private):", psi_deployment$psi$PSI[1], "\n")
psi_deployment$pic

# 9. EXPORT RESULTS
# ------------------------------------------------------------------------------
df_private$Score <- private_scores$score
write.csv(df_private %>% select(id, Score), "final_private_results.csv", row.names=FALSE)
cat("\nScript Complete. Flexible Cohort + Macro Vars implemented successfully.\n")
