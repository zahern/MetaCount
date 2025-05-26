library(dplyr)
library(tidyr)
library(ggplot2)
library(lme4)
library(car)
library(glmmTMB)
# Load gravity dataset

current_dir_set()
# Load gravity_gc dataset (replace with the actual path or URL)
gravity_gc <- read.csv('Alterynx merged dataset_v3.csv')

# Add new columns to gravity_gc
gravity_gc$year <- 2018
gravity_gc$sqrt_distance <- sqrt(gravity_gc$Avg_Road_OP_Dist + 1)
gravity_gc$ln_distance <- log(gravity_gc$Avg_Road_OP_Dist + 1)
gravity_gc$ln_pop <- log(gravity_gc$POPHH + 1)
gravity_gc$Inc <- log(gravity_gc$`Avg Medium Income` + 1)
gravity_gc$one <- 1

# Replace infinite values with NA
gravity_gc <- gravity_gc %>%
  mutate(across(everything(), ~ ifelse(is.infinite(.), NA, .)))

# Check for missing values
print(colSums(is.na(gravity_gc)))

# Check specific columns for missing values
print(colSums(is.na(gravity_gc %>% select(Ticket, Avg_Road_OP_Dist, `Avg Medium Income`, Avg_PT_OP_Time, Avg_Road_OP_Time, Accommodation_Rooms))))

# Ensure dependent variable is numeric
gravity_gc$Ticket <- as.numeric(gravity_gc$Ticket)

# Add dummy variables for fixed effects (Sport)
sports_dummies <- model.matrix(~ Sport - 1, data = gravity_gc)
sports_dummies <- sports_dummies[, !grepl("SportSwimming", colnames(sports_dummies))] # Drop "Sport_Swimming"

# Combine independent variables and fixed effects
gravity_gc <- cbind(gravity_gc, sports_dummies)

# Fit a Poisson GLMM with random intercept for Sport
poisson_glmm <- glmer(
  Ticket ~ sqrt_distance + `Avg.Medium.Income` + ln_pop + (1 | SA2_NAME21),
  data = gravity_gc,
  family = poisson(link = "log")
)

# Print model summary
summary(poisson_glmm)

# Add predicted values to the dataset
gravity_gc$predicted <- predict(poisson_glmm, type = "response")

# Plot observed vs predicted counts
a <- ggplot(gravity_gc, aes(x = Ticket, y = predicted)) +
  geom_point(alpha = 0.6, color = "blue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(
    x = "Observed Counts",
    y = "Predicted Counts",
    title = "Observed vs. Predicted Counts (GLMM)"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom") 
a

ggsave("mixed.png")



gravity_gc <- gravity_gc %>% arrange(Avg_Road_OP_Dist)

gravity_gc$INC <- gravity_gc$Avg.Medium.Income
gravity_gc$Denom_SQRT.DIST <- gravity_gc$Avg.Medium.Income
nb_glmm <- glmmTMB(
  Ticket ~ sqrt_distance + `Avg.Medium.Income` + ln_pop + (1 | SA2_NAME21),
  data = gravity_gc,
  family = nbinom2(link = "log")  # Negative Binomial (Type 2)
)

nb_glmm <- glmmTMB(
  Ticket ~ sqrt_distance + `Avg.Medium.Income` + ln_pop + (1 | SA2_NAME21),
  zi=~sqrt_distance,
  data = gravity_gc,
  family = truncated_nbinom2(link = "log")  # Negative Binomial (Type 2)
)

nb_glmm <- glmmTMB(
  Ticket ~ Avg_Road_OP_Dist+POPHH + `Avg.Medium.Income`,
  data = gravity_gc,
  family = nbinom2(link = "log")  # Negative Binomial (Type 2)
  )
  
  

nb_glmm_s <- glmmTMB(
  Ticket ~ sqrt(Avg_Road_OP_Dist)+POPHH + `Avg.Medium.Income`,
  data = gravity_gc,
  family = nbinom2(link = "log"),  # Negative Binomial (Type 2))
  
)

nb_glmm_st <- glmmTMB(
  Ticket ~ sqrt(Avg_Road_OP_Dist)+POPHH + `Avg.Medium.Income`+ (1 | Sport),
  data = gravity_gc,
  family = nbinom2(link = "log"),  # Negative Binomial (Type 2))
  
)
summary(nb_glmm_st)
summary(nb_glmm_s)
# Print model summary
summary(nb_glmm)


gravity_gc$POP.INC_per_SQRT.DIST <- gravity_gc$POPHH*gravity_gc$Avg.Medium.Income/sqrt(gravity_gc$Avg_Road_OP_Dist)
gravity_gc$POP_per_SQRT.DIST <- gravity_gc$POPHH/sqrt(gravity_gc$Avg_Road_OP_Dist)
gravity_gc$POP.INC_EMP_per_SQRT.DIST <- (gravity_gc$POPHH*gravity_gc$Avg.Medium.Income+gravity_gc$Total_Emp)/sqrt(gravity_gc$Avg_Road_OP_Dist)
gravity_gc$POP.INC_DEPC_per_SQRT.DIST <- (gravity_gc$POPHH*gravity_gc$Avg.Medium.Income+gravity_gc$DepC)/sqrt(gravity_gc$Avg_Road_OP_Dist)
gravity_gc$POP_per_DIST<- gravity_gc$POPHH/(gravity_gc$Avg_Road_OP_Dist)

old_o <- lm(
  Ticket ~ POP.INC_per_SQRT.DIST,
  data = gravity_gc,  # Negative Binomial (Type 2))
  
)

old_p <- lm(
  Ticket ~ POP_per_SQRT.DIST,
  data = gravity_gc,  # Negative Binomial (Type 2))

)
old_q <- lm(
  Ticket ~ POP.INC_EMP_per_SQRT.DIST,
  data = gravity_gc,  # Negative Binomial (Type 2))
  
)
old_r <- lm(
  Ticket ~ POP.INC_DEPC_per_SQRT.DIST,
  data = gravity_gc,  # Negative Binomial (Type 2))
  
)
old_s <- lm(
  Ticket ~ POP_per_DIST,
  data = gravity_gc,  # Negative Binomial (Type 2))
  
)


# Add predicted values to the dataset
#these are the predicted
gravity_gc$predicted <- predict(nb_glmm, type = "response")
gravity_gc$predicted_sqrt <-  predict(nb_glmm_s, type = "response")
gravity_gc$predicted_mm <-  predict(nb_glmm_st, type = "response")
gravity_gc$pred_o <- predict(old_o, type = "response")
gravity_gc$pred_p <- predict(old_p, type = "response")
gravity_gc$pred_q <- predict(old_q, type = "response")
gravity_gc$pred_r <- predict(old_r, type = "response")
gravity_gc$pred_s <- predict(old_s, type = "response")
predict_data$cum_o <- cumsum(gravity_gc$pred_o)
predict_data$cum_p <- cumsum(gravity_gc$pred_p)
predict_data$cum_q <- cumsum(gravity_gc$pred_q)
predict_data$cum_r <- cumsum(gravity_gc$pred_r)
predict_data$cum_s <- cumsum(gravity_gc$pred_s)

#gravity_gc$Ticket
#this is the actual
gravity_gc$Ticket

predict_data <- gravity_gc
predict_data$Dspark <- cumsum(gravity_gc$Ticket)
predict_data$cumulative_tickets <- cumsum(gravity_gc$predicted)
predict_data$cumulative_tickets_sqrt <- cumsum(gravity_gc$predicted_sqrt)
predict_data$cumulative_tickets_mm <- cumsum(gravity_gc$predicted_mm)
predict_data$cumulative_proportion <- predict_data$cumulative_tickets / max(predict_data$cumulative_tickets)
predict_data$distance_values <- gravity_gc$Avg_Road_OP_Dist
b <- ggplot(predict_data, aes(x = distance_values, y = cumulative_proportion)) +
  geom_line(color = "blue", size = 1.2) +
  labs(
    title = "Cumulative Trip Length Distribution for Tickets Sold",
    x = "Distance",
    y = "Cumulative Proportion of Tickets Sold"
  ) +
  theme_minimal()
b

bb <- ggplot(predict_data, aes(x = distance_values)) +
  #geom_line(color = "blue", size = 1.2) +
  # Cumulative Tickets
  geom_line(aes(y = cumulative_tickets, color = "NB Distance Model"), size = 1, linetype = 'dashed') +
  geom_line(aes(y = cumulative_tickets_sqrt, color = "NB Sqrt Distance Model"), size = 1, linetype = 'dashed') +
  geom_line(aes(y = cumulative_tickets_mm, color = "Mixed Model (Random Effects)"), size = 1, linetype = 'dashed') +
  geom_line(aes(y = Dspark, color ="Tickets Sold"), size = 1) +
  labs(
    title = "Cumulative Trip Length Distribution for Tickets Sold",
    x = "Distance",
    y = "Cumulative Proportion of Tickets Sold"
  ) +
  theme_minimal()
bb


cc <- ggplot(predict_data, aes(x = distance_values)) +
  #geom_line(color = "blue", size = 1.2) +
  # Cumulative Tickets
  geom_line(aes(y = cumulative_tickets, color = "NB_Dist"), size = 1, linetype = 'dashed') +
  geom_line(aes(y = cumulative_tickets_sqrt, color = "NB_sqrt_dist"), size = 1, linetype = 'dashed') +
  geom_line(aes(y = cumulative_tickets_mm, color = "NB-RE"), size = 1, linetype = 'dashed') +
  geom_line(aes(y = cum_o, color = "TMR_ff1"), size = 1, linetype = 'twodash') +
  geom_line(aes(y = cum_p, color = "TMR_ff2"), size = 1, linetype = 'twodash') +
  geom_line(aes(y = cum_q, color = "TMR_ff3"), size = 1, linetype = 'twodash') +
  geom_line(aes(y = cum_r, color = "TMR_ff4"), size = 1, linetype = 'twodash') +
  geom_line(aes(y = cum_s, color = "TMR_ff5"), size = 1, linetype = 'twodash') +
  geom_line(aes(y = Dspark, color ="Tickets Sold"), size = 1) +
  labs(
    title = "Cumulative Trip Length Distribution for Tickets Sold",
    x = "Distance",
    y = "Cumulative Proportion of Tickets Sold"
  ) + scale_color_manual(name = "Models")+
  theme_minimal()

cc


cc <- ggplot(predict_data, aes(x = distance_values)) +
  # Cumulative Tickets with different colors and line types
  geom_line(aes(y = cumulative_tickets, color = "NB Distance Model", linetype = "NB Distance Model"), size = 1) +
  geom_line(aes(y = cumulative_tickets_sqrt, color = "NB Sqrt Distance Model", linetype = "NB Sqrt Distance Model"), size = 1) +
  geom_line(aes(y = cumulative_tickets_mm, color = "Mixed Model (Random Effects)", linetype = "Mixed Model (Random Effects)"), size = 1) +
  geom_line(aes(y = cum_o, color = "Pop*Inc/sqrt(dist)", linetype = "Pop*Inc/sqrt(dist)"), size = 1) +
  geom_line(aes(y = cum_p, color = "Pop/sqrt(dist)", linetype = "Pop/sqrt(dist)"), size = 1) +
  geom_line(aes(y = cum_q, color = "Pop*Inc+Emp/sqrt(dist)", linetype = "Pop*Inc/sqrt(dist)"), size = 1) +
  geom_line(aes(y = cum_r, color = "Pop*Inc+DepC/sqrt(dist)", linetype = "Pop*Inc+DepC/sqrt(dist)"), size = 1) +
  geom_line(aes(y = cum_s, color = "Pop/dist", linetype = "Pop/dist"), size = 1) +
  geom_line(aes(y = Dspark, color = "Tickets Sold", linetype = "Tickets Sold"), size = 1) +
  # Manual color scale
  scale_color_manual(
    name = "Models",
    values = c(
      "NB_Dist" = "blue",
      "NB_sqrt_distce Model" = "cyan",
      "Mixed Model (Random Effects)" = "lightblue",
      "Pop*Inc/sqrt(dist)" = "purple",
      "Pop/sqrt(dist)" = "lavender",
      "Pop*Inc+Emp/sqrt(dist)" = "maroon",
      "Pop*Inc+DepC/sqrt(dist)" = "hotpink",
      "Pop/dist" = "pink",
      "Tickets Sold" = "black"
    )
  ) +
  # Manual line type scale
  scale_linetype_manual(
    name = "Models",
    values = c(
      "NB Distance Model" = "dotted",
      "NB Sqrt Distance Model" = "dotted",
      "Mixed Model (Random Effects)" = "dotted",
      "Pop*Inc/sqrt(dist)" = "twodash",
      "Pop/sqrt(dist)" = "twodash",
      "Pop*Inc+DepC/sqrt(dist)" = "twodash",
      "Pop/dist" = "twodash",
      "Tickets Sold" = "solid"
    )
  ) +
  guides(
    linetype = guide_legend("Models")
  ) +
  
  # Add labels and minimal theme
  labs(
    title = "Cumulative Trip Length Distribution for Tickets Sold",
    x = "Distance",
    y = "Cumulative Proportion of Tickets Sold"
  ) +
  theme_minimal()

# Print the plot
cc
print(cc)


# Create a new column in your data if needed, or just use labels directly
cc <- ggplot(predict_data, aes(x = distance_values)) +
  geom_line(aes(y = cumulative_tickets, group = "NB_dist", color = "NB_dist", linetype = "NB_dist"), size = 1) +
  geom_line(aes(y = cumulative_tickets_sqrt, group = "NB_sqrt_dist", color = "NB_sqrt_dist", linetype = "NB_sqrt_dist"), size = 1) +
  geom_line(aes(y = cumulative_tickets_mm, group = "NB-RE", color = "NB-RE", linetype = "NB-RE"), size = 1) +
  geom_line(aes(y = cum_o, group = "TMR_ff1", color = "TMR_ff1", linetype = "TMR_ff1"), size = 1) +
  geom_line(aes(y = cum_p, group = "TMR_ff2", color = "TMR_ff2", linetype = "TMR_ff2"), size = 1) +
  geom_line(aes(y = cum_q, group = "TMR_ff3", color = "TMR_ff3", linetype = "TMR_ff3"), size = 1) +
  geom_line(aes(y = cum_r, group = "TMR_ff4", color = "TMR_ff4", linetype = "TMR_ff4"), size = 1) +
  geom_line(aes(y = cum_s, group = "TMR_ff5", color = "TMR_ff5", linetype = "TMR_ff5"), size = 1) +
  geom_line(aes(y = Dspark, group = "Tickets Sold", color = "Tickets Sold", linetype = "Tickets Sold"), size = 1) +
  scale_color_manual(
    name = "Models",
    values = c(
      "NB_dist" = "blue",
      "NB_sqrt_dist" = "cyan",
      "NB-RE" = "lightblue",
      "TMR_ff1" = "purple",
      "TMR_ff2" = "lavender",
      "TMR_ff3" = "maroon",
      "TMR_ff4" = "hotpink",
      "TMR_ff5" = "pink",
      "Tickets Sold" = "black"
    )
  ) +
  scale_linetype_manual(
    name = "Models",
    values = c(
      "NB_dist" = "dotted",
      "NB_sqrt_dist" = "dotted",
      "NB-RE" = "dotted",
      "TMR_ff1" = "twodash",
      "TMR_ff2" = "twodash",
      "TMR_ff3" = "twodash",
      "TMR_ff4" = "twodash",
      "TMR_ff5" = "twodash",
      "Tickets Sold" = "solid"
    )
  ) +
  labs(
    title = "Cumulative Trip Length Distribution for Tickets Sold",
    x = "Distance",
    y = "Cumulative Proportion of Tickets Sold"
  ) +
  theme_minimal()


cc











ggsave("cumulative_plot.png", plot = cc, width = 10, height = 6, dpi = 300)


dd <- ggplot(gravity_gc) +
  # Plot predicted vs actual for each model
  geom_point(aes(x = predicted, y = Ticket, color = "NB_Dist"), alpha = 0.6) +
  geom_point(aes(x = predicted_sqrt, y = Ticket, color = "NB_sqrt_dist"), alpha = 0.6) +
  geom_point(aes(x = predicted_mm, y = Ticket, color = "NB-RE"), alpha = 0.6) +
  geom_point(aes(x = pred_o, y = Ticket, color = "TMR_ff1"), alpha = 0.6) +
  geom_point(aes(x = pred_p, y = Ticket, color = "TMR_ff2"), alpha = 0.6) +
  geom_point(aes(x = pred_q, y = Ticket, color = "TMR_ff3"), alpha = 0.6) +
  geom_point(aes(x = pred_r, y = Ticket, color = "TMR_ff4"), alpha = 0.6) +
  geom_point(aes(x = pred_s, y = Ticket, color = "TMR_ff5"), alpha = 0.6) +
  # Add a reference line (y = x) to indicate perfect prediction
  geom_abline(intercept = 0, slope = 1, color = "black", linetype = "dashed", size = 1) +
  # Scale for colors
  scale_color_manual(
    name = "Models",
    values = c(
      "NB_Dist" = "blue",
      "NB_sqrt_dist" = "cyan",
      "NB-RE" = "lightblue",
      "TMR_ff1" = "purple",
      "TMR_ff2" = "lavender",
      "TMR_ff3" = "maroon",
      "TMR_ff4" = "hotpink",
      "TMR_ff5" = "pink"
    )
  ) +
  # Add labels and title
  labs(
    title = "Predicted vs Actual Tickets",
    x = "Predicted Tickets",
    y = "Actual Tickets"
  ) +
  # Minimal theme for clean visuals
  theme_minimal()

# View the plot
dd

long_data <- gravity_gc %>%
  select(Ticket, predicted, predicted_sqrt, predicted_mm, pred_o, pred_p, pred_q, pred_r, pred_s) %>%
  pivot_longer(
    cols = starts_with("pred"),  # Columns to pivot (predicted model columns)
    names_to = "Model",          # New column for model names
    values_to = "Predicted"      # New column for predicted values
  )

# Rename the model names for better labels
long_data$Model <- factor(long_data$Model, levels = c(
  "predicted", "predicted_sqrt", "predicted_mm", "pred_o", "pred_p", "pred_q", "pred_r", "pred_s"
), labels = c(
  "NB_Dist", "NB_sqrt_dist", "NB-RE", "TMR_ff1", "TMR_ff2", "TMR_ff3", "TMR_ff4", "TMR_ff5"
))

# Create the faceted plot
faceted_plot <- ggplot(long_data, aes(x = Predicted, y = Ticket)) +
  geom_point(alpha = 0.6, color = "blue") +  # Scatter plot for predicted vs actual
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black", size = 1) +  # Perfect prediction line
  facet_wrap(~Model, scales = "free", ncol = 2) +  # Create facets for each model
  labs(
    title = "Predicted vs Actual Tickets by Model",
    x = "Predicted Tickets",
    y = "Actual Tickets"
  ) +
  theme_minimal() +
  theme(
    strip.text = element_text(size = 10, face = "bold"),  # Styling for facet labels
    plot.title = element_text(hjust = 0.5)               # Center the plot title
  )

# View the plot
faceted_plot
ggsave("faceted_plota.png", plot = dd, width = 21, height = 21, dpi = 300)
ggsave("faceted_plot.png", plot = faceted_plot, width = 21, height = 30, dpi = 600)




model_colors <- c(
  "NB_Dist" = "blue",
  "NB_sqrt_dist" = "cyan",
  "NB-RE" = "lightblue",
  "TMR_ff1" = "purple",
  "TMR_ff2" = "lavender",
  "TMR_ff3" = "maroon",
  "TMR_ff4" = "hotpink",
  "TMR_ff5" = "pink"
)

# Create the faceted plot with model-specific colors
faceted_plot <- ggplot(long_data, aes(x = Predicted, y = Ticket, color = Model)) +
  geom_point(alpha = 0.6) +  # Scatter plot for predicted vs actual
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black", size = 1) +  # Perfect prediction line
  facet_wrap(~Model, scales = "free", ncol = 2) +  # Create facets for each model
  scale_color_manual(values = model_colors) +  # Assign custom colors to models
  labs(
    title = "Predicted vs Actual Tickets by Model",
    x = "Predicted Tickets",
    y = "Actual Tickets"
  ) +
  theme_minimal() +
  theme(
    strip.text = element_text(size = 10, face = "bold"),  # Styling for facet labels
    plot.title = element_text(hjust = 0.5),              # Center the plot title
    legend.position = "none"                             # Remove legend for clarity in facets
  )

# View the faceted plot
print(faceted_plot)

# Save each facet as a separate PNG file
facet_models <- unique(long_data$Model)  # Get the list of model names
output_dir <- "facets"  # Directory to save PNG files
dir.create(output_dir, showWarnings = FALSE)  # Create directory if it doesn't exist

for (model in facet_models) {
  # Filter data for the current model
  model_data <- long_data %>% filter(Model == model)
  
  # Create a plot for the current model
  single_facet_plot <- ggplot(model_data, aes(x = Predicted, y = Ticket)) +
    geom_point(alpha = 0.6, color = model_colors[as.character(model)]) +  # Model-specific color
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black", size = 1) +
    labs(
      title = paste("Predicted vs Actual Tickets -", model),
      x = "Predicted Tickets",
      y = "Actual Tickets"
    ) +
    theme_minimal()
  
  # Save the plot as a PNG file
  ggsave(
    filename = file.path(output_dir, paste0(model, ".png")),
    plot = single_facet_plot,
    width = 6, height = 4, dpi = 300
  )
}







library(broom)
library(performance)
library(dplyr)
#install.packages("broom.mixed")
library(broom.mixed)

extract_logLik <- function(model) {
  tryCatch(as.numeric(logLik(model)), error = function(e) NA)
}

# Function to safely calculate RMSE
extract_rmse <- function(model) {
  tryCatch(performance::performance_rmse(model), error = function(e) NA)
}
logLik_nb_glmm <- extract_logLik(nb_glmm)
logLik_nb_glmm_s <- extract_logLik(nb_glmm_s)  # Replace with your model objects
logLik_nb_glmm_st <- extract_logLik(nb_glmm_st)
logLik_old_o <- extract_logLik(old_o)
logLik_old_p <- extract_logLik(old_p)
logLik_old_q <- extract_logLik(old_q)
logLik_old_r <- extract_logLik(old_r)
logLik_old_s <- extract_logLik(old_s)

rmse_nb_glmm <- extract_rmse(nb_glmm)
rmse_nb_glmm_s <- extract_rmse(nb_glmm_s)  # Replace with your model objects
rmse_nb_glmm_st <- extract_rmse(nb_glmm_st)
rmse_old_o <- extract_rmse(old_o)
rmse_old_p <- extract_rmse(old_p)
rmse_old_q <- extract_rmse(old_q)
rmse_old_r <- extract_rmse(old_r)
rmse_old_s <- extract_rmse(old_s)

# List of models
models <- list(
  nb_glmm = nb_glmm,
  nb_glmm_s = nb_glmm_s,
  nb_glmm_st = nb_glmm_st,
  old_o = old_o,
  old_p = old_p,
  old_q = old_q,
  old_r = old_r,
  old_s = old_s
)


# Extract coefficients
coef_table <- lapply(models, broom.mixed::tidy) %>%
  bind_rows(.id = "Model") %>%
  select(Model, term, estimate)

# Goodness-of-fit metrics: Log-likelihood and RMSE
fit_metrics <- lapply(models, function(model) {
  data.frame(
    Model = deparse(substitute(model)),
    LogLik = as.numeric(logLik(model)),
    RMSE = performance::performance_rmse(model)
  )
}) %>%
  bind_rows()

# Combine coefficients and fit metrics
final_table <- coef_table %>%
  left_join(fit_metrics, by = "Model")

# Print the table
print(final_table)





model_metrics <- tibble(
  Model = c("nb_glmm", "nb_glmm_s", "nb_glmm_st", "old_o", "old_p", "old_q", "old_r", "old_s"),
  LogLik = c(logLik_nb_glmm, logLik_nb_glmm_s, logLik_nb_glmm_st, logLik_old_o, logLik_old_p, logLik_old_q, logLik_old_q, logLik_old_s),
  RMSE = c(rmse_nb_glmm, rmse_nb_glmm_s, rmse_nb_glmm_st, rmse_old_o, rmse_old_p, rmse_old_q, rmse_old_r, rmse_old_s)
)




# Coefficients (example)
coef_table <- bind_rows(
  broom.mixed::tidy(nb_glmm) %>% mutate(Model = "nb_glmm"),
  broom.mixed::tidy(nb_glmm_s) %>% mutate(Model = "nb_glmm_s"),
  broom.mixed::tidy(nb_glmm_st) %>% mutate(Model = "nb_glmm_st"),
  broom::tidy(old_o) %>% mutate(Model = "old_o"),
  broom::tidy(old_p) %>% mutate(Model = "old_p"),
  broom::tidy(old_q) %>% mutate(Model = "old_q"),
  broom::tidy(old_r) %>% mutate(Model = "old_r"),
  broom::tidy(old_s) %>% mutate(Model = "old_s")
)

# Merge coefficients with metrics
final_table <- coef_table %>%
  left_join(model_metrics, by = "Model")

# View final table
print(final_table)


# Add LogLik as a new row for each model
reshaped_table <- final_table %>%
  select(Model, term, estimate, LogLik) %>% # Keep relevant columns
  distinct(Model, LogLik) %>%              # Get unique LogLik per model
  mutate(term = "LogLik") %>%              # Add "LogLik" to the term column
  bind_rows(
    final_table %>% select(Model, term, estimate) # Add original terms and estimates
  ) %>%
  pivot_wider(
    names_from = Model,                     # Use 'Model' as new column headers
    values_from = estimate                  # Use 'estimate' for row values
  )

# View the reshaped table
print(reshaped_table)
#pivot table
pivoted_table <- final_table %>%
  select(Model, term, estimate) %>% # Keep only relevant columns
  pivot_wider(
    names_from = Model,             # Use 'Model' column for new column names
    values_from = estimate          # Use 'estimate' column for values
  )

# View the reshaped table
print(pivoted_table)


process_model_terms <- function(pivoted_table) {
  # Define the mapping between terms and their corresponding model functions
  term_to_model <- c(
    "POP.INC_per_SQRT.DIST" = "function",
    "POP_per_SQRT.DIST" = "function",
    "POP.INC_EMP_per_SQRT.DIST" = "function",
    "POP.INC_DEPC_per_SQRT.DIST" = "function",
    "POP_per_DIST" = "function"
  )
  
  # Replace terms with corresponding model function names, retain others
  updated_table <- pivoted_table %>%
    mutate(
      term = case_when(
        term %in% names(term_to_model) ~ term_to_model[term], # Replace mapped terms
        TRUE ~ term                   # Retain other terms
      )
    )
  
  # Consolidate rows: group by `term` and compact numeric columns
  consolidated_table <- updated_table %>%
    group_by(term) %>%
    summarise(
      across(
        where(is.numeric), 
        ~ if (all(is.na(.))) NA_real_ else first(na.omit(.))
      ), 
      .groups = "drop"
    )
  
  # Return the consolidated table
  return(consolidated_table)
}
  
  # Return the updated table


# Example usage
processed_table <- process_model_terms(pivoted_table)

# View the result
print(processed_table)
# Example usage
processed_table <- process_model_terms(pivoted_table)
renamed_table <- processed_table %>%
  rename(
    Term = term,                      # Rename `term` to `Term`
    NB_Dist = nb_glmm,                 # Rename `nb_glmm` to `NB_GLM`
    NB_sqrt_Dist = nb_glmm_s,             # Rename `nb_glmm_s` to `NB_GLM_S`
    MM = nb_glmm_st,           # Rename `nb_glmm_st` to `NB_GLM_ST`
    o1 = old_o,                    # Rename `old_o` to `Old_O`
    o2 = old_p,                    # Rename `old_p` to `Old_P`
    o3 = old_q,                    # Rename `old_q` to `Old_Q`
    o4 = old_r,                    # Rename `old_r` to `Old_R`
    o5 = old_s                     # Rename `old_s` to `Old_S`
  )
# View the result
print(renamed_table)
renamed_table <- renamed_table %>%
  mutate(
    Term = case_when(
      Term == 'sd__(Intercept)'~ 'sd (Venue)',
      Term == 'Avg.Medium.Income'~ 'Income',
      Term == 'Avg_Road_OP_Dist'~ 'Distance',
      Term == 'sqrt(Avg_Road_OP_Dist)'~ 'sqrt(Distance)',
      Term == 'POPHH'~'Pop',
      TRUE ~ Term
    )
  )
renamed_table

renamed_model_metrics <- model_metrics %>%
  mutate(
    Model = case_when(
      Model == "nb_glmm" ~ "NB_Dist",        # Rename `nb_glmm` to `NB_Dist`
      Model == "nb_glmm_s" ~ "NB_sqrt_Dist", # Rename `nb_glmm_s` to `NB_sqrt_dist`
      Model == "nb_glmm_st" ~ "MM",          # Rename `nb_glmm_st` to `MM`
      Model == "old_o" ~ "o1",               # Rename `old_o` to `o1`
      Model == "old_p" ~ "o2",               # Rename `old_p` to `o2`
      Model == "old_q" ~ "o3",
      Model == "old_r" ~ "o4",
      Model == "old_s" ~ "o5",
      TRUE ~ Model                           # Keep other names unchanged
    )
  )


renamed_model_metrics
library(gt)
library(officer)
library(flextable)




styled_tablemm <- renamed_model_metrics %>%
  gt() %>%
  tab_style(
    style = list(cell_fill(color = "blue")),    # Light blue background
    locations = cells_body(
      rows = Model == "NB_Dist"                     # Highlight rows where Model is "NB_Dist"
    )
  ) %>%
  tab_style(
    style = list(cell_fill(color = "cyan")),  # Light green background
    locations = cells_body(
      rows = Model == "NB_sqrt_dist"                # Highlight rows where Model is "NB_sqrt_dist"
    )) %>%
      tab_style(
        style = list(cell_fill(color = "lightblue")),  # Light green background
        locations = cells_body(
          rows = Model == "MM"                # Highlight rows where Model is "NB_sqrt_dist"
      )
   ) %>%
  tab_style(
    style = list(cell_fill(color = "purple")),  # Purple background
    locations = cells_body(
      rows = Model == "o1"  # Highlight rows where Model is "o1"
    )
  ) %>%
  tab_style(
    style = list(cell_fill(color = "lavender")),  # Lavender background
    locations = cells_body(
      rows = Model == "o2"  # Highlight rows where Model is "o2"
    )
  ) %>%
  tab_style(
    style = list(cell_fill(color = "maroon")),  # Maroon background
    locations = cells_body(
      rows = Model == "o3"  # Highlight rows where Model is "o3"
    )
  ) %>%
  tab_style(
    style = list(cell_fill(color = "hotpink")),  # Hot Pink background
    locations = cells_body(
      rows = Model == "o4"  # Highlight rows where Model is "o4"
    )
  ) %>%
  tab_style(
    style = list(cell_fill(color = "pink")),  # Pink background
    locations = cells_body(
      rows = Model == "o5"  # Highlight rows where Model is "o5"
    )
  ) %>%
  tab_header(
    title = "Model Metrics Table",
    subtitle = "Models Compared"
  )

# View the styled table
styled_tablemm %>% as_flextable()

styled_flextable <- as_word(styled_tablemm)

# Create a Word document and add the table
doc <- read_docx() %>%
  body_add_flextable(value = styled_flextable) %>%
  body_add_par("Model Metrics Table", style = "heading 1")

# Save the Word document
print(doc, target = "styled_lemm.docx")

# Save the styled table as an HTML file
gtsave(styled_tablemm, "styled_temm.docx")

model_description <- data.frame(
  Model = c("NB_Dist", "NB_sqrt_dist", "MM", "o1", "o2", "o3", "o4", "o5"),
  Description = c(
    "Negative Binomial: Distance, Population and Income",
    "Negative Binomial: Square Root Distance, Population and Income",
    "Negative Nibomial Random Effects Model on Venue:Square Root Distance, Population and Income",
    "Population x Income | denom = Square Root Distance",
    "Population | denom = Square Root Distance",
    "Population x Income +Employment | denom = Square Root Distance",
    "Population x Income Dep C | denom = Square Root Distance",
    "Population | denom =  Distance"
  ),
  Color = c("blue", "cyan", "lightblue", "purple", "lavender", "maroon", "hotpink", "pink")
)
model_description

tyled_table <- model_description %>%
  gt() %>%
  tab_style(
    style = list(cell_fill(color = "blue")),  # Blue background
    locations = cells_body(
      rows = Model == "NB_Dist"  # Highlight rows where Model is "NB_Dist"
    )
  ) %>%
  tab_style(
    style = list(cell_fill(color = "cyan")),  # Cyan background
    locations = cells_body(
      rows = Model == "NB_sqrt_dist"  # Highlight rows where Model is "NB_sqrt_dist"
    )
  ) %>%
  tab_style(
    style = list(cell_fill(color = "lightblue")),  # Light blue background
    locations = cells_body(
      rows = Model == "MM"  # Highlight rows where Model is "MM"
    )
  ) %>%
  tab_style(
    style = list(cell_fill(color = "purple")),  # Purple background
    locations = cells_body(
      rows = Model == "o1"  # Highlight rows where Model is "o1"
    )
  ) %>%
  tab_style(
    style = list(cell_fill(color = "lavender")),  # Lavender background
    locations = cells_body(
      rows = Model == "o2"  # Highlight rows where Model is "o2"
    )
  ) %>%
  tab_style(
    style = list(cell_fill(color = "maroon")),  # Maroon background
    locations = cells_body(
      rows = Model == "o3"  # Highlight rows where Model is "o3"
    )
  ) %>%
  tab_style(
    style = list(cell_fill(color = "hotpink")),  # Hot pink background
    locations = cells_body(
      rows = Model == "o4"  # Highlight rows where Model is "o4"
    )
  ) %>%
  tab_style(
    style = list(cell_fill(color = "pink")),  # Pink background
    locations = cells_body(
      rows = Model == "o5"  # Highlight rows where Model is "o5"
    )
  ) %>%
  tab_header(
    title = "Model Description Table",
    subtitle = "Models with Embedded Colors"
  )

gtsave(tyled_table, 'tt.docx')
gtsave(model_description, "styled_tabldemm.docx")

# Use the processed_table or renamed_table as input

styled_table <- renamed_table %>%
  gt() %>%
  
  fmt_scientific(
    columns = c(NB_Dist, NB_sqrt_Dist, MM, o1, o2, o3, o4, o5),
    decimals = 2 # Number of digits after the decimal point in the mantissa
  ) %>%
  tab_style(
    style = list(cell_fill(color = "blue")),
    locations = cells_body(columns = NB_Dist)
  ) %>%
  tab_style(
    style = list(cell_fill(color = "cyan")),
    locations = cells_body(columns = NB_sqrt_Dist)
  ) %>%
  tab_style(
    style = list(cell_fill(color = "lightblue")),
    locations = cells_body(columns = MM)
  ) %>%
  tab_style(
    style = list(cell_fill(color = "purple")),
    locations = cells_body(columns = o1)
  ) %>%
  tab_style(
    style = list(cell_fill(color = "lavender")),
    locations = cells_body(columns = o2)
  ) %>%
  tab_style(
    style = list(cell_fill(color = "maroon")),
    locations = cells_body(columns = o3)
  ) %>%
  tab_style(
    style = list(cell_fill(color = "hotpink")),
    locations = cells_body(columns = o4)
  ) %>%
  tab_style(
    style = list(cell_fill(color = "pink")),
    locations = cells_body(columns = o5)
  ) %>%
  tab_header(
    title = "Comparison of Models",
    subtitle = "Proposed Models vs Report Models"
  )


# View the styled table
styled_table
gtsave(styled_table, 'fdff.docx')

# Plot observed vs predicted counts
a <- ggplot(gravity_gc, aes(x = Ticket, y = predicted)) +
  geom_point(alpha = 0.6, color = "blue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(
    x = "Observed Counts",
    y = "Predicted Counts",
    title = "Observed vs. Predicted Counts (Negative Binomial GLMM)"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom") 
a
ggsave("negative_binomial_glmm.png")
