library(glmnet)

args <- commandArgs(trailingOnly = TRUE)

train_path  <- args[1]
test_path   <- args[2]
output_path <- args[3]
vars_string <- args[4]
offset_name <- args[5]

vars <- unlist(strsplit(vars_string, ","))

train <- read.csv(train_path)
test  <- read.csv(test_path)

# Remove NA rows
train <- na.omit(train)
test  <- na.omit(test)

# Fix offset (must be > 0)
train[[offset_name]][train[[offset_name]] <= 0] <- 1e-6
test[[offset_name]][test[[offset_name]] <= 0] <- 1e-6

# Stabilize glmnet on Windows
glmnet::glmnet.control(factory = TRUE)

# Build model matrices
X_train <- model.matrix(~ . - 1, data=train[, vars])
X_test  <- model.matrix(~ . - 1, data=test[, vars])

y_train <- train$Y
offset_train <- train[[offset_name]]
offset_test  <- test[[offset_name]]

# Fit Poisson LASSO with offset
fit <- cv.glmnet(
  x = X_train,
  y = y_train,
  family = "poisson",
  offset = offset_train,
  alpha = 1,
  standardize = FALSE
)

# Extract coefficients at lambda.min
coef_table <- as.matrix(coef(fit, s = "lambda.min"))
coef_df <- data.frame(
  variable = rownames(coef_table),
  coefficient = coef_table[,1]
)

# Keep only coefficients >= 0.01 in magnitude
coef_df <- coef_df[abs(coef_df$coefficient) >= 0.15, ]

print("===== Variables After Thresholding (>= 0.01) =====")
print(coef_df)

# Build refit formula (exclude intercept)
selected_vars <- coef_df$variable[coef_df$variable != "(Intercept)"]

if (length(selected_vars) == 0) {
  # No variables survived thresholding → fallback to intercept-only model
  refit_formula <- as.formula(paste("Y ~ 1 + offset(", offset_name, ")"))
} else {
  refit_formula <- as.formula(
    paste("Y ~", paste(selected_vars, collapse=" + "),
          "+ offset(", offset_name, ")")
  )
}

print("===== Refit Formula =====")
print(refit_formula)

# Refit Poisson GLM with selected variables
refit <- glm(refit_formula, data=train, family=poisson)

# Compute train BIC
train_bic <- BIC(refit)

# Predict on test
preds <- predict(refit, newdata=test, type="response")
rmse <- sqrt(mean((test$Y - preds)^2))

# Save output
write.csv(
  data.frame(bic = train_bic, rmse = rmse),
  output_path,
  row.names = FALSE
)