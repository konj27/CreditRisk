# MPE_APRE
# Project: Mortgage PD Scorecard
# Authors: Vojtech Grebenicek, Jan Konarik

# 0. ENVIRONMENT SETUP  ---------------------------------------------------
# Clear workspace
rm(list = ls())

# Load (and install necessary packages if missing)
if (!require(scorecard)) install.packages("scorecard")
if (!require(tidyverse)) install.packages("tidyverse")
if (!require(caret)) install.packages("caret")
if (!require(pROC)) install.packages("pROC")
if (!require(car)) install.packages("car")
if (!require(zoo)) install.packages("zoo")
if (!require(corrplot)) install.packages("corrplot")
if (!require(ggplot2)) install.packages("ggplot2")
if (!require(glmnet)) install.packages("glmnet")

library(scorecard)
library(tidyverse)
library(caret)
library(pROC)
library(car)
library(zoo)
library(corrplot)
library(ggplot2)
library(glmnet)

# 1. DATA IMPORT AND PREPARATION --------------------------------------------
# Load the dataset (from the Working Directory)
data_raw <- read.csv("mortgage_sample.csv")

# Separation of Samples
# Public: Model development (Train/Test)
# Private: Final blind validation (Deployment check)
df_public  <- data_raw %>% filter(sample == "public")
df_private <- data_raw %>% filter(sample == "private")

# Create Target (Public only)
# Define custom target function:
# a. If default is found anywhere in window -> Return 1
# b. If default is not found and window is < 12 months -> Return NA
# c. If default is not found and window is 12 months -> Return 0
calc_target_12m <- function(x) {
    if (any(x == 1, na.rm = TRUE)) {
        return(1)
    } else if (length(x) < 12) {
        return(NA)
    } else {
        return(0)
    }
}

# Sort by ID/Time and look forward for default in next 12 months (if possible)
df_cohort <- df_public %>%
    arrange(id, time) %>%
    group_by(id) %>%
    mutate(
        # Look for default, using defined custom target function for complete
        # 12 month windows and partial windows (if there are less observations)
        target_12m = rollapply(default_time, width = 12, FUN = calc_target_12m,
                               align = "left", fill = NA, partial = TRUE),
        # Calculate months since start
        t_from_start = time - first_time
    ) %>%
    # Filter for flexible cohorts
    # Keep the first observation (t_from_start = 0) and then every 12 months
    filter((t_from_start %% 12) == 0) %>%
    filter(!is.na(target_12m)) %>% # Remove rows with incomplete future
    ungroup() %>%
    select(-t_from_start) # Clean up helper columns

# 2. VARIABLE SELECTION ---------------------------------------------------
# Define target variable
target_var <- "target_12m"

# Exclude technical/non-predictor variables and define inputs
vars_exclude <- c("id", "time", "orig_time", "first_time", "mat_time",
                  "default_time", "payoff_time", "status_time", "sample")

inputs <- setdiff(names(df_cohort), c(vars_exclude, target_var))

# Check for missing values (remove variable if > 50 % is missing)
missing_rates <- sapply(df_cohort[inputs], function(x) sum(is.na(x)) / length(x))
vars_missing <- names(missing_rates[missing_rates > 0.5])

if (length(vars_missing) > 0) {
    cat("\nDropping variables with > 50 % missing:", paste(vars_missing, collapse = ", "), "\n")
    inputs <- setdiff(inputs, vars_missing)
}

# 3. SPLITTING & BINNING (WOE)  -------------------------------------------

# ===========
# Time-based split (70 % Train / 30 % Test)

#time_cutoff <- quantile(df_cohort$time, 0.7)
#df_train <- df_cohort %>% filter(time <= time_cutoff)
#df_test  <- df_cohort %>% filter(time > time_cutoff)
# ===========

# Random Split 70/30 =======
train_index <- createDataPartition(df_cohort[[target_var]], p = 0.7, list = FALSE)

df_train <- df_cohort[train_index, ]
df_test  <- df_cohort[-train_index, ]
# ========

# Distribution on Target variable
table(df_train[[target_var]])
prop.table(table(df_train[[target_var]]))
ggplot(df_train, aes(x = .data[[target_var]])) +
    geom_bar(fill = "steelblue") +
    labs(title = "Distribution of Target Variable",
         y = "Count",
         x = "Target Class") +
    theme_minimal()

# WoE Binning (Handles outliers and missing values automatically)
bins <- woebin(df_train, y = target_var, x = inputs, min_perc_total = 0.05)

# WoE Transformation
train_woe <- woebin_ply(df_train, bins)
test_woe  <- woebin_ply(df_test, bins)

# 4. UNIVARIATE AND MULTIVARIATE CHECKS -----------------------------------
# IV selection (> 0.02)
iv_values <- iv(df_train, y = target_var, x = inputs)
strong_vars <- iv_values %>% filter(info_value > 0.02) %>% pull(variable)

cat("\nVariables kept (IV > 0.02):", length(strong_vars), "\n")
print(strong_vars)

# Correlation check (Threshold > 0.5, Prevents multicolinearity)
strong_vars_woe <- paste0(strong_vars, "_woe")
vars_to_check <- intersect(names(train_woe), strong_vars_woe)
cor_mat <- cor(train_woe %>% select(all_of(vars_to_check)), method = "spearman")
corrplot(cor_mat)

# Find high correlations
high_corr_idx <- findCorrelation(cor_mat, cutoff = 0.5)
high_corr_vars_woe <- vars_to_check[high_corr_idx]

# Identify final variables (Keep the '_woe' version for modeling)
final_vars_woe <- setdiff(vars_to_check, high_corr_vars_woe)
final_vars_woe
cat("Dropped due to Correlation > 0.5:", paste(high_corr_vars_woe, collapse = ", "), "\n")

# VIF check
form_vif <- as.formula(paste(target_var, "~", paste(final_vars_woe, collapse = " + ")))
m_vif_check <- glm(form_vif, data = train_woe, family = binomial())

vif_vals <- car::vif(m_vif_check)
print(vif_vals)

if (any(vif_vals > 5)) cat("WARNING: High VIF (>5) detected.\n") else cat("VIF Check Passed.\n")

# 5. MODELLING (STEPWISE LOGISTIC REGRESSION) -----------------------------
# Prepare final datasets with Target + Selected WoE variables
train_final <- train_woe %>% select(all_of(c(target_var, final_vars_woe)))
test_final  <- test_woe  %>% select(all_of(c(target_var, final_vars_woe)))

m_full <- glm(as.formula(paste(target_var, "~ .")), family = binomial(), data = train_final)
m_step <- step(m_full, direction = "both", trace = 0)

summary(m_step)

# 6. SCORECARD SCALING ----------------------------------------------------
# Parameters: Base = 500, Odds = 1:50, PDO = 50
card <- scorecard(bins, m_step,
                  points0 = 500,
                  odds0   = 1/50,
                  pdo     = 50)

# Calculate scores
train_score <- scorecard_ply(df_train, card)
test_score  <- scorecard_ply(df_test, card)

# Print points
print(card[[1]])

# 7. PERFORMANCE AND DEPLOYMENT CHECK -------------------------------------
pred_train <- predict(m_step, newdata = train_final, type = "response")
pred_test  <- predict(m_step, newdata = test_final, type = "response")

# ROC and GINI
roc_train <- roc(df_train[[target_var]], pred_train)
roc_test  <- roc(df_test[[target_var]], pred_test)
gini_train <- (2 * auc(roc_train) - 1) * 100
gini_test  <- (2 * auc(roc_test) - 1) * 100

cat(sprintf("\nPERFORMANCE:\nTrain GINI: %.2f%%\nTest GINI:  %.2f%%\n", gini_train, gini_test))

# ROC plot
plot(roc_train, col = "blue", main = "ROC Curve: Train vs Test", lwd = 2)
lines(roc_test, col = "red", lwd = 2)
legend("bottomright", legend = c("Train", "Test"), col = c("blue", "red"), lwd = 2)

# Deployment PSI (Public vs Private)
# Private data must be WoE transformed using scorecard_ply
private_scores <- scorecard_ply(df_private, card)

psi_deployment <- perf_psi(
    score = list(train = train_score, private = private_scores)
)

cat("\nDeployment PSI (Public vs Private):", psi_deployment$psi$psi[1], "\n")
psi_deployment$pic

# 9. EXPORT RESULTS -------------------------------------------------------
df_private$Score <- private_scores$score
write.csv(df_private %>% select(id, Score), "final_private_results.csv", row.names = FALSE)
cat("\nResults saved to 'final_private_results.csv'.\n")

# -------------------------------------------------------------------------

# 10. DEMONSTRATION: LASSO MODEL COMPARISON -------------------------------
# Run LASSO
x_matrix <- as.matrix(train_final %>% select(-all_of(target_var)))
y_vector <- train_final[[target_var]]

lasso_cv <- cv.glmnet(x_matrix, y_vector, family = "binomial", alpha = 1, type.measure = "auc")
coef_lasso <- coef(lasso_cv, s = "lambda.1se")

# Extract LASSO variables
lasso_vars <- rownames(coef_lasso)[which(coef_lasso != 0)]
lasso_vars <- lasso_vars[lasso_vars != "(Intercept)"]

# Refit GLM for LASSO (to get comparable GINI)
if(length(lasso_vars) > 0) {
    form_lasso <- as.formula(paste(target_var, "~", paste(lasso_vars, collapse = " + ")))
    m_lasso <- glm(form_lasso, family = binomial(), data = train_final)

    # Calculate LASSO Performance
    pred_lasso_test <- predict(m_lasso, newdata = test_final, type="response")
    roc_lasso <- roc(test_final[[target_var]], pred_lasso_test, quiet = TRUE)
    gini_lasso <- (2 * auc(roc_lasso) - 1) * 100
} else {
    gini_lasso <- 0
    m_lasso <- NULL
}

# Comparison table
comparison <- data.frame(
    Metric = c("Variables Selected", "Test GINI"),
    Stepwise_Main = c(length(coef(m_step))-1, sprintf("%.2f%%", gini_test)),
    Lasso_Challenger = c(length(lasso_vars), sprintf("%.2f%%", gini_lasso))
)
print(comparison)

# Plot comparison ROC
plot(roc_test, col = "blue", lwd = 2, main = "ROC Comparison: Stepwise vs LASSO")
if (!is.null(m_lasso)) lines(roc_lasso, col = "red", lwd = 2, lty = 2)
legend("bottomright", legend = c("Stepwise (Main)", "LASSO (Challenger)"),
       col = c("blue", "red"), lwd = 2, lty = 1:2)
