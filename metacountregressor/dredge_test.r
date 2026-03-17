library(MASS)
library(MuMIn)
options(na.action = "na.fail")

args <- commandArgs(trailingOnly = TRUE)

train_path  <- args[1]
test_path   <- args[2]
output_path <- args[3]
vars_string <- args[4]
offset_name <- args[5]

vars <- unlist(strsplit(vars_string, ","))

train <- read.csv(train_path)
test  <- read.csv(test_path)

# Keep only needed columns
train <- train[, c("Y", vars, offset_name)]
test  <- test[, c("Y", vars, offset_name)]

# Build formula
formula_string <- paste(
  "Y ~",
  paste(vars, collapse = " + "),
  "+ offset(", offset_name, ")"
)

form <- as.formula(formula_string)
print(form)

# Fit full negative binomial model
full_model <- glm.nb(form, data = train)

# Run dredge
results <- dredge(full_model, rank = "BIC", subset = dc(w <= 3))

# Extract best model
best_model <- get.models(results, 1)[[1]]
print(summary(best_model))

# Compute train BIC
train_bic <- BIC(best_model)

# Predict on test set
preds <- predict(best_model, newdata = test, type = "response")
rmse <- sqrt(mean((test$Y - preds)^2))

# Save results
write.csv(
  data.frame(bic = train_bic, rmse = rmse),
  output_path,
  row.names = FALSE
)