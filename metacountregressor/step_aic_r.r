library(MASS)

args <- commandArgs(trailingOnly = TRUE)

train_path  <- args[1]
test_path   <- args[2]
output_path <- args[3]
vars_string <- args[4]
offset_name <- args[5]

vars <- unlist(strsplit(vars_string, ","))

train <- read.csv(train_path)
test  <- read.csv(test_path)

# Build formula
formula_string <- paste(
  "Y ~",
  paste(vars, collapse = " + "),
  "+ offset(", offset_name, ")"
)

form <- as.formula(formula_string)

# Fit full NB model
full <- glm.nb(form, data=train)

# Stepwise AIC selection
best <- stepAIC(full, direction="both", trace=FALSE)
print(best)
print('bic')
print(BIC(best))
# Predict
preds <- predict(best, newdata=test, type="response")

# RMSE
rmse <- sqrt(mean((test$Y - preds)^2))

# Save output
write.csv(
  data.frame(bic = BIC(best), rmse = rmse),
  output_path,
  row.names = FALSE
)