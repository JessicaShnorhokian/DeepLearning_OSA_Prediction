library(RColorBrewer)
library(tidyverse)

# Set the path to your CSV file
path <- "data/OSA_complete_patients.csv"

# Read the data
df <- read_csv(path)

# Drop the first column
df <- df %>% select(-1)

# Display the first few rows
head(df, 5)

# Add columns AHI5, AHI15, and AHI30
df <- df %>%
  mutate(
    AHI_5 = if_else(Severity >= 1, 1, 0),
    AHI_15 = if_else(Severity >= 2, 1, 0),
    AHI_30 = if_else(Severity >= 3, 1, 0)
  )

# Display the updated dataframe
head(df, 5)

cutoff_val <- 10

# Initialize empty lists
numerical_columns <- c()
categorical_columns <- c()

# Identify column types
for (column_name in names(df)) {
  unique_vals <- n_distinct(df[[column_name]])
  # Check if the column is categorical based on unique values and data type
  if (is.character(df[[column_name]]) || unique_vals <= cutoff_val) {
    categorical_columns <- c(categorical_columns, column_name)
  } else {
    numerical_columns <- c(numerical_columns, column_name)
  }
}

# Ensure the output directory exists
dir.create("stat_results", showWarnings = FALSE)

# Perform Chi-Square Tests for categorical columns
for(column in categorical_columns){
  # Create contingency table
  contingency_table <- table(df$Severity, df[[column]])
  
  # Save contingency table to CSV

  #If pvlaue is less than 0.05, sabve the column
  
  write.csv(contingency_table, file = paste0("stat_results/contingency_table_", column, ".csv"))
  
  # Perform Chi-Square Test
  chi_square_result <- chisq.test(contingency_table)
  
  # Extract results to save
  chi_square_summary <- tibble(
    Statistic = chi_square_result$statistic,
    PValue = chi_square_result$p.value,
    Method = chi_square_result$method
  )
  if (chi_square_result$p.value < 0.05) {
  write.csv(contingency_table, file = paste0("stat_results/contingency_table_", column, ".csv"))
  }
  
  # Flatten the expected values matrix to a data frame
  expected_df <- as.data.frame(as.table(chi_square_result$expected))
  names(expected_df) <- c("Row", "Column", "Expected")
  
  # Save chi-square test summary to CSV
  write.csv(chi_square_summary, file = paste0("stat_results/chi_square_result_", column, ".csv"), row.names = FALSE)
  
  # Save expected values to CSV
  write.csv(expected_df, file = paste0("stat_results/expected_values_", column, ".csv"), row.names = FALSE)
  
  # Print chi-square test result
  print(chi_square_result)
}



for (column in numerical_columns) {
  # Perform Kruskal-Wallis Test
  kruskal_test <- kruskal.test(df[[column]] ~ df$Severity)
  
  # Save Kruskal-Wallis Test results to CSV
  kruskal_summary <- tibble(
    Statistic = kruskal_test$statistic,
    PValue = kruskal_test$p.value
  )
  
  # Save Kruskal-Wallis Test results to CSV

  write.csv(kruskal_summary, file = paste0("stat_results/kruskal_result_", column, ".csv"), row.names = FALSE)
  
  # Print Kruskal-Wallis Test results
  print(kruskal_test)
}

significant_columns <- c()
sig_columns_sig_pvalue <- c()
tests <- c()


#check wich csvs are significant
files <- list.files("stat_results", pattern = "chi_square_result", full.names = TRUE)
for (file in files) {
  data <- read.csv(file)
  if (data$PValue < 0.05) {
    significant_columns <- c(significant_columns, gsub("stat_results/chi_square_result_", "", gsub(".csv", "", file)))
    sig_columns_sig_pvalue <- c(sig_columns_sig_pvalue, data$PValue)
    tests <- c(tests, "Chi-Square")
  

  }
}

files_kruskal <- list.files("stat_results", pattern = "kruskal_result", full.names = TRUE)
for (file in files_kruskal) {
  data <- read.csv(file)
  if (data$PValue < 0.05) {
    significant_columns <- c(significant_columns, gsub("stat_results/kruskal_result_", "", gsub(".csv", "", file)))
    sig_columns_sig_pvalue <- c(sig_columns_sig_pvalue, data$PValue)
    tests <- c(tests, "Kruskal-Wallis")

  }
}

#save significant columns and corresponding pvalues to a file
significant_columns_df <- tibble(
  Column = significant_columns,
  PValue = sig_columns_sig_pvalue,
  test = tests
)

write.csv(significant_columns_df, file = "stat_results/significant_columns.csv", row.names = FALSE)
