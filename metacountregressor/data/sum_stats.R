# Load necessary libraries
library(tidyverse)  # for data manipulation and summarization
library(xtable)     # for LaTeX table output
library(knitr)
# Set working directory to the script's directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Set the path to the CSV file
file_path <- "rural_int.csv"

# Read the dataset
data <- read.csv(file_path)
data
data <- data %>%
  filter(PEAKHR >-1)
  

data <- subset(data, select = -c(town, weather_station, county))
data <- subset(data, select = -c(weather_station, county))
data
# View the first few rows of the dataset to understand its structure
print(head(data))

# Summary statistics for the 'FREQ' column
freq_summary <- summary(data$crashes)
print(freq_summary)

# Converting the summary to a data frame for better handling in xtable
freq_summary_df <- data.frame(Statistic = names(freq_summary), Value = as.vector(freq_summary))

# Create LaTeX code for the frequency summary
latex_freq_summary <- xtable(freq_summary_df, caption = "Summary Statistics for Frequency", label = "tab:freq_summary")
print(latex_freq_summary, type = "latex", include.rownames = FALSE)

# Assuming other columns are categorical and represent contributing factors
contributing_factors <- names(data)[names(data) != "crashes"]  # Adjust based on your actual data structure

# Loop through each factor and print a LaTeX table summary
for (factor in contributing_factors) {
  factor_summary <- summary(as.factor(data[[factor]]))
  factor_summary_df <- data.frame(Level = names(factor_summary), Count = as.vector(factor_summary))
  latex_table <- xtable(factor_summary_df, caption = paste("Summary for", factor), label = paste("tab:", factor))
  
  print(latex_table, type = "latex", include.rownames = FALSE)
}

# Optionally, explore correlations or other statistical tests
# Example placeholder: replace 'numeric_factor' with the actual column name if applicable
# cor(data$FREQ, data$numeric_factor, use = "complete.obs")  # Only if 'numeric_factor' is indeed numeric

# Save the LaTeX tables to a file
sink("latex_summary_output.tex")
print(latex_freq_summary, type = "latex", include.rownames = FALSE)
for (factor in contributing_factors) {
  factor_summary <- summary(as.factor(data[[factor]]))
  factor_summary_df <- data.frame(Level = names(factor_summary), Count = as.vector(factor_summary))
  latex_table <- xtable(factor_summary_df, caption = paste("Summary for", factor), label = paste("tab:", factor))
  
  print(latex_table, type = "latex", include.rownames = FALSE)
}
sink()



# Summary function for continuous and categorical variables
summarize_data <- function(data, var) {
  if (is.numeric(data[[var]])) {
    # Check if the range is between 0 to 5 or similar and treat as categorical
    if (all(data[[var]] %in% 0:5)) {
      return(c("Type" = "Categorical", "Categories" = toString(unique(data[[var]]))))
    } else {
      return(c("Type" = "Continuous", "Range" = paste(min(data[[var]]), "to", max(data[[var]]))))
    }
  } else {
    return(c("Type" = "Categorical", "Categories" = toString(unique(data[[var]]))))
  }
}

# Using lapply to apply the function across chosen variables
summary_list <- lapply(names(data), function(x) summarize_data(data, x))
summary_list
# Convert the list to a data frame
summary_df <- do.call(rbind, summary_list)
rownames(summary_df) <- names(data)

# Create a LaTeX table
summary_df
kable(summary_df, format = "latex", booktabs = TRUE, caption = "Summary of Contributing Factors")

