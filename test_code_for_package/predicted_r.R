# Load the data
data <- read.csv("1848.csv")

# Fit a Poisson GLM to the data
model <- glm(FSI ~ US + S, data = data, family = poisson)

# View the model summary
summary(model)
