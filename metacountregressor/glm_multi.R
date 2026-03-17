library(glmulti)
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

# ✅ Keep only needed columns
train <- train[, c("Y", vars, offset_name)]
test  <- test[, c("Y", vars, offset_name)]

# ✅ Build formula with offset
formula_string <- paste(
  "Y ~",
  paste(vars, collapse = " + "),
  "+ offset(", offset_name, ")"
)

form <- as.formula(formula_string)

print(form)

result <- glmulti(
  form,
  data = train,
  level = 1,
  method = "g",
  crit = "bic",
  fitfunction = "glm.nb"
)

best_model <- result@objects[[1]]
print(summary(best_model))

train_bic <- BIC(best_model)

preds <- predict(best_model, newdata=test, type="response")
rmse <- sqrt(mean((test$Y - preds)^2))

write.csv(
  data.frame(bic=train_bic, rmse=rmse),
  output_path,
  row.names=FALSE
)