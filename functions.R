# -------------------------------------------------------------
# 3. OUTLIER DETECTION
# -------------------------------------------------------------


detect_outliers <- function(data, z_threshold = 3) {
  numeric_vars <- names(data)[sapply(data, is.numeric)]
  
  outlier_summary <- sapply(numeric_vars, function(var) {
    vals <- data[[var]]
    z_out <- sum(abs(scale(vals)) > z_threshold, na.rm = TRUE)
    iqr_out <- sum(vals < quantile(vals, 0.25) - 1.5 * IQR(vals) |
                     vals > quantile(vals, 0.75) + 1.5 * IQR(vals),
                   na.rm = TRUE)
    c(z_score = z_out,
      iqr = iqr_out,
      pct_iqr = round(iqr_out / length(vals) * 100, 1))
  }) %>% t() %>% as.data.frame()
  
  print(outlier_summary)
  
  # Boxplots
  data_long <- pivot_longer(data[numeric_vars], cols = everything(),
                            names_to = "variable", values_to = "value")
  
  p <- ggplot(data_long, aes(x = variable, y = value)) +
    geom_boxplot(fill = "lightblue", outlier.color = "red") +
    coord_flip() +
    theme_minimal() +
    labs(title = "Outlier Detection", x = NULL, y = NULL)
  
  print(p)
}

# -------------------------------------------------------------
# 4. OUTLIER TREATMENT (Capping using IQR)
# -------------------------------------------------------------
cap_outliers <- function(data, vars) {
  data %>% 
    mutate(across(all_of(vars), ~ {
      q1 <- quantile(.x, 0.25, na.rm = TRUE)
      q3 <- quantile(.x, 0.75, na.rm = TRUE)
      iqr_val <- q3 - q1
      lower <- q1 - 1.5 * iqr_val
      upper <- q3 + 1.5 * iqr_val
      pmax(lower, pmin(.x, upper))
    }))
}

# -------------------------------------------------------------
# 5. CORRELATION ANALYSIS
# -------------------------------------------------------------

plot_correlation <- function(data, threshold = 0.5) {
  
  # Compute correlation matrix
  cor_matrix <- data %>% 
    select(where(is.numeric)) %>% 
    cor(use = "complete.obs")
  
  # Updated correlation plot (upper triangle)
  p <- ggcorrplot(
    cor_matrix,
    method = "square",
    type = "lower",                   # use upper triangle
    lab = TRUE,                       # show correlation values
    digits = 2,                       # round correlation numbers
    lab_size = 3.5,
    outline.color = "white",
    hc.order = TRUE,                  # cluster similar variables
    colors = c("#E46726", "white", "#3A78B7"),  # red → white → blue
    title = "Correlation Matrix",
    ggtheme = theme_minimal()
  ) +
    theme(
      plot.title = element_text(hjust = 0.5, size = 15, face = "bold"),
      axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
      axis.text.y = element_text(size = 10)
    )
  
  print(p)
  
  # High correlation reporting
  cor_df <- as.data.frame(as.table(cor_matrix)) %>% 
    filter(abs(Freq) > threshold, Var1 != Var2) %>% 
    arrange(desc(abs(Freq)))
  
  if (nrow(cor_df) > 0) {
    cat("\nHigh correlations (|r| >", threshold, "):\n")
    print(cor_df, row.names = FALSE)
  }
}

# -------------------------------------------------------------
# 6. DISTRIBUTIONS & BOX PLOTS BY DIABETES STATUS
# -------------------------------------------------------------
plot_distributions_by_diabetes <- function(data) {
  data %>% 
    mutate(diabetes = factor(Diabetes, labels = c("No", "Yes"))) %>% 
    select(where(is.numeric), diabetes) %>% 
    pivot_longer(-diabetes, names_to = "variable", values_to = "value") %>% 
    ggplot(aes(x = value, fill = diabetes)) +
    geom_density(alpha = 0.6) +
    facet_wrap(~variable, scales = "free", ncol = 4) +
    scale_fill_manual(values = c("#6D9EC1", "#E46726")) +
    theme_minimal() +
    labs(title = "Distribution by Diabetes Status", fill = "Diabetes")
}

plot_boxplots_by_diabetes <- function(data) {
  data %>% 
    mutate(diabetes = factor(Diabetes, labels = c("No", "Yes"))) %>% 
    select(where(is.numeric), diabetes) %>% 
    pivot_longer(-diabetes, names_to = "variable", values_to = "value") %>% 
    ggplot(aes(x = diabetes, y = value, fill = diabetes)) +
    geom_boxplot(alpha = 0.7) +
    facet_wrap(~variable, scales = "free", ncol = 4) +
    scale_fill_manual(values = c("#6D9EC1", "#E46726")) +
    theme_minimal() +
    labs(title = "Variables by Diabetes Status", x = NULL, fill = "Diabetes")
}


# -------------------------------------------------------------
# 7. SUMMARY STATISTICS BY DIABETES CLASS
# -------------------------------------------------------------
summary_by_diabetes <- function(data) {
  data %>% 
    mutate(diabetes = factor(Diabetes, labels = c("No", "Yes"))) %>% 
    group_by(diabetes) %>% 
    summarise(across(where(is.numeric),
                     list(mean = ~mean(.x, na.rm = TRUE),
                          median = ~median(.x, na.rm = TRUE)),
                     .names = "{.col}_{.fn}")) %>% 
    pivot_longer(-diabetes, names_to = "metric", values_to = "value") %>% 
    separate(metric, into = c("variable", "stat"), sep = "_(?=[^_]+$)") %>% 
    pivot_wider(names_from = stat, values_from = value) %>% 
    arrange(variable)
}

cap_outliers <- function(data, vars) {
  data %>% 
    mutate(across(all_of(vars), ~ {
      q1 <- quantile(.x, 0.25, na.rm = TRUE)
      q3 <- quantile(.x, 0.75, na.rm = TRUE)
      iqr_val <- q3 - q1
      lower <- q1 - 1.5 * iqr_val
      upper <- q3 + 1.5 * iqr_val
      pmax(lower, pmin(.x, upper))
    }))
}


predict_diabetes_sentence <- function(new_data, model = final_fit) {
  # Ensure input is a data frame
  new_data <- as.data.frame(new_data)
  
  # Align factor levels with training data
  factor_vars <- c("Gender", "Smoking", "Alcohol", "FamilyHistory")
  for (var in factor_vars) {
    if (var %in% colnames(new_data)) {
      new_data[[var]] <- factor(new_data[[var]], levels = levels(train[[var]]))
    }
  }
  
  # Predict probabilities
  preds <- predict(model, new_data, type = "prob")
  
  # Extract probability of diabetes
  prob_yes <- round(preds$.pred_Yes * 100, 1)
  
  # Determine predicted class
  pred_class <- ifelse(prob_yes > 50, "has", "does not have")
  
  # Create sentence
  sentence <- paste0("Based on the patient's data, the model predicts that the patient ", 
                     pred_class, " diabetes with a probability of ", prob_yes, "%.")
  
  return(sentence)
}