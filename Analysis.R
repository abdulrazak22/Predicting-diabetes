###############################################################
# DATA SCIENCE FOUNDATIONS – END-TO-END PROJECT TEMPLATE (R)
###############################################################

# -------------------------------------------------------------
# 0. Install & Load Required Packages
# -------------------------------------------------------------
packages <- c("tidyverse", "caret", "ggplot2", "corrplot", "caTools",
              "randomForest", "e1071", "ROCR", "ggcorrplot")

install.packages(setdiff(packages, installed.packages()[,1]))

library(tidyverse)
library(caret)
library(corrplot)
library(caTools)
library(randomForest)
library(e1071)
library(ROCR)
library(ggcorrplot)
# Install tidymodels if not installed
if (!require("tidymodels")) install.packages("tidymodels")
library(tidymodels)
library(doParallel)
library("ranger")
library("kernlab")
library("xgboost")

# -------------------------------------------------------------
# 1. Import Dataset
# -------------------------------------------------------------
data <- read.csv("health_dataset_diabetes.csv")
glimpse(data)

# -------------------------------------------------------------
# 2. Missing Values & Data Types
# -------------------------------------------------------------
colSums(is.na(data))                 # Missing values per column
sum(is.na(data))                     # Total missing
(sum(is.na(data)) / (nrow(data) * ncol(data))) * 100

data <- data %>%
  mutate(
    bmi = ifelse(is.na(bmi), median(bmi, na.rm = TRUE), bmi),
    blood_pressure = ifelse(is.na(blood_pressure), median(blood_pressure, na.rm = TRUE), blood_pressure)
  ) %>% 
  rename(
    Age = age,
    Gender = gender,
    BMI = bmi,
    BloodPressure = blood_pressure,
    Glucose = glucose_level,
    Cholesterol = cholesterol,
    ExerciseHrs = exercise_hours_per_week,
    Smoking = smoking_status,
    Alcohol = alcohol_intake,
    FamilyHistory = family_history,
    SleepHrs = sleep_hours_per_night,
    Stress = stress_level,
    WaistCM = waist_circumference_cm,
    HeartRate = heart_rate_bpm,
    Diabetes = diabetes
  )

colSums(is.na(data))


# Convert all character columns to factors
data <- data %>% mutate(across(where(is.character), as.factor))

# Identify numeric columns
numeric_vars <- names(data)[sapply(data, is.numeric)]
numeric_vars

# Run outlier detection
detect_outliers(data)
detect_outliers(data, z_threshold = 2.5)

data_processed <- cap_outliers(data, 
                               c("Cholesterol", "BloodPressure",
                                 "WaistCM", "BMI"))
# Compare before/after
summary(data[c("Cholesterol", "BMI")])
summary(data_processed[c("Cholesterol", "BMI")])

plot_correlation(data_processed)
plot_correlation(data_processed, threshold = 0.3)


# Run plots
plot_distributions_by_diabetes(data_processed)
plot_boxplots_by_diabetes(data_processed)

summary_stats <- summary_by_diabetes(data_processed)
print(summary_stats)

data_processed <- cap_outliers(
  data,
  c("Cholesterol", "BloodPressure", "WaistCM", "BMI")
)

summary(data[c("Cholesterol", "BMI")])
summary(data_processed[c("Cholesterol", "BMI")])

summary(data_processed)

set.seed(123)

data_processed <- data_processed %>%
  mutate(
    # Target as factor with proper levels
    Diabetes = factor(Diabetes, levels = c(0,1), labels = c("No","Yes")),
    
    # Make sure categorical predictors are factors
    Gender = factor(Gender),
    Smoking = factor(Smoking),
    Alcohol = factor(Alcohol),
    FamilyHistory = factor(FamilyHistory),
    
    # Force numeric predictors to numeric
    across(c(Age, BMI, BloodPressure, Glucose, Cholesterol,
             ExerciseHrs, SleepHrs, Stress, WaistCM, HeartRate),
           as.numeric)
  )

split <- initial_split(data_processed, prop = 0.8, strata = Diabetes)
train <- training(split)
test  <- testing(split)


recipe_diabetes <- recipe(Diabetes ~ ., data = train) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())


model_log <- logistic_reg(mode = "classification") %>%
  set_engine("glm")

model_rf <- rand_forest(
  trees = 500,
  mtry = tune(),
  min_n = tune()
) %>% 
  set_engine("ranger") %>%
  set_mode("classification")


model_svm <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune()
) %>% 
  set_engine("kernlab") %>%
  set_mode("classification")


model_xgb <- boost_tree(
  trees = 600,
  learn_rate = tune(),
  tree_depth = tune()
) %>% 
  set_engine("xgboost") %>%
  set_mode("classification")


models <- workflow_set(
  preproc = list(diabetes_recipe = recipe_diabetes),
  models = list(
    logistic = model_log,
    random_forest = model_rf,
    svm = model_svm,
    xgboost = model_xgb
  )
)

###############################################################
# FINAL MODEL SELECTION + PENALIZED LOGISTIC REGRESSION FIXED
###############################################################

# 5-fold CV
cv_folds <- vfold_cv(train, v = 5, strata = Diabetes)

registerDoParallel()

# Run all models through workflow_map()
model_results <- workflow_map(
  models,
  resamples = cv_folds,
  metrics = metric_set(accuracy, roc_auc),
  grid = 20,
  verbose = TRUE
)

# Rank models by best ROC-AUC (or accuracy)
rank_results <- model_results %>% 
  rank_results(select_best = TRUE) %>%
  arrange(desc(mean))

rank_results

# Best model name (character)
best_model_name <- rank_results$wflow_id[1]
best_model_name

# 1. Extract the best tuning results for the chosen model
best_params <- model_results %>%
  extract_workflow_set_result(best_model_name) %>%
  select_best(metric = "roc_auc")  # or "accuracy" if you prefer

# 2. Finalize the workflow with these best parameters
final_wflow <- final_wflow %>%
  finalize_workflow(best_params)

# 3. Fit the workflow on the full training data
final_fit <- final_wflow %>%
  fit(data = train)


# 1. Generate predictions and bind with truth
final_predictions <- predict(final_fit, test, type = "prob") %>%
  bind_cols(predict(final_fit, test)) %>%
  bind_cols(test %>% select(Diabetes))


# 4. Compute confusion matrix
conf_mat(final_predictions, truth = Diabetes, estimate = .pred_class)

metrics <- final_predictions %>% 
  metrics(truth = Diabetes, estimate = .pred_class)

# metrics


roc_data <- roc_curve(final_predictions, truth = Diabetes, .pred_Yes)

auc_val <- roc_auc(final_predictions, truth = Diabetes, .pred_Yes)$.estimate

ggplot(roc_data, aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(color = "blue", size = 1) +
  geom_abline(lty = 2, color = "gray") +
  labs(
    title = paste0("ROC Curve (AUC = ", round(auc_val, 3), ")"),
    x = "1 - Specificity",
    y = "Sensitivity"
  ) +
  theme_minimal()

cm <- final_predictions %>%
  conf_mat(truth = Diabetes, estimate = .pred_class)

roc_data <- final_predictions %>%
  roc_curve(truth = Diabetes, .pred_Yes)


# Plot
cm_plot <- autoplot(cm, type = "heatmap") + ggtitle("Confusion Matrix")
roc_plot <- autoplot(roc_data) + ggtitle("ROC Curve")


new_patient <- tibble(
  Age = 55,
  Gender = "Male",
  BMI = 26,
  BloodPressure = 134,
  Glucose = 110,
  Cholesterol = 195,
  ExerciseHrs = 2,
  Smoking = "Never",
  Alcohol = "Light",
  FamilyHistory = "Yes",
  SleepHrs = 7,
  Stress = 6,
  WaistCM = 82,
  HeartRate = 75
)

predict_diabetes_sentence(new_patient)
