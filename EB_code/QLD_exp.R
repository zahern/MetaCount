# Header ------------------------------------------------------------------

# Fitting a beta linear regression in JAGS
# Andrew Parnell and Ahmed Ali

# In this code we generate some data from a beta linear regression model and fit is using jags. We then intepret the output.

# Some boiler plate code to clear the workspace, and load in required packages
rm(list = ls()) # Clear the workspace
library(R2jags)
library(boot)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y_t = repsonse variable for observation t=1,..,N - should be in the range (0, 1)
# x_t = explanatory variable for obs t
# alpha, beta = intercept and slope parameters to be estimated
# sigma = residual standard deviation

# Likelihood:
# y_t ~ Beta(a[t], b[t])
# mu[t] = a[t]/(a[t] + b[t])
# a[t] = mu[t] * phi
# b[t] = (1 - mu[t]) * phi
# logit(mu[t]) = alpha + beta * x[t]
# Prior
# alpha ~ N(0,100) - vague priors
# beta ~ N(0,100)
# phi ~ U(0, 100)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
T <- 100
alpha <- -1
beta <- 0.2
phi <- 5
# Set the seed so this is repeatable
set.seed(123)
#x <- sort(runif(T, 0, 10)) # Sort as it makes the plotted lines neater
data = read.csv('data/Stage5A.csv')
#logit_mu <- alpha + beta * x
#mu <- inv.logit(logit_mu)
#a <- mu * phi
#b <- (1 - mu) * phi
#y <- rbeta(T, a, b)

y <- data$crashes
x1 <- data$SW
x2 <- data$US
x3 <- data$RSHS
x4 <- log(data$AADT)
x5 <- data$VS_CURVE
#id_reg <- as.numeric(as.factor(data$county))
off <- log(data$LEN_YR *1000)/10
#np.log(x_df['LEN_YR'] * 1000) / 10  # Offset
#x2 <- data$paved_shoulder
#lnF <- log(data$FAADT)
#z1 <- data$DP01
#z3 <- data$DX32

# Also creat a plot
#plot(x, y)
#lines(x, mu)

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data

model_code <- "
model {
  for(i in 1:N) {
    ## Likelihood
    y[i] ~ dnegbin(prob[i], phi)
    prob[i] <- phi / (phi + eps[i] * mu[i])
    log(mu[i]) <- b2 * x2[i] + b3 * x6[i] + b4 * x7[i] + b5 * off[i]

    eps[i] ~ dgamma(f[i], t)  # Use a single t for all i
    f[i] <- 1 + z[i]
    z[i] ~ dbern(k)  # Use a single k for all i

    ## Log-likelihood
    LL[i] <- (loggam(phi + y[i]) - loggam(phi) - loggam(y[i] + 1)) +
              (phi) * (log(phi) - log(eps[i] * mu[i] + phi)) + 
              y[i] * (log(eps[i] * mu[i]) - log(eps[i] * mu[i] + phi))
  }      

  ## Priors for k and t
  k ~ dunif(0, 1)
  t <- (1 - k) / k
  mean.eps <- (t + 2) / (t * (t + 1))
  log.mean.eps <- log(mean.eps)
  for(j in 1:200) {
    
    b1_reg[j] ~ dnorm(b1_reg_bar, b1_reg_tau)
  }
  ## Priors for regression coefficients
  b1 ~ dnorm(0, 0.1)
  b2 ~ dnorm(0, 0.1)
  b3 ~ dnorm(0, 0.1)
  b4 ~ dnorm(0, 0.1)
  b5 ~ dnorm(0, 0.1)

  ## Priors for phi
  phi ~ dgamma(0.1, 0.1)

  ## Calculate b0 based on the mean values of eps
  b0 <- log.mean.eps - b1 * 8.979 - b2 * 69.159 - b3 * 14.288 - b4 * 0.291 - b5 * (-1.246)
}"

# Set up the data
model_data <- list(N= 37080, y = y, lnF = lnF, z1 = z1, z3=z3, x2 = x2, x6=x6, x7 =x7, off = off, id_reg = id_reg)

# Choose the parameters to watch
model_parameters <- c("z", "eps", "phi", "b5", "b4", "b3", "b2")

# Run the model
model_run <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code),n.chains = 4,
  n.iter = 1000,
  n.burnin = 200,
  n.thin = 2
)

# Simulated results -------------------------------------------------------

# Check the output - are the true values inside the 95% CI?
# Also look at the R-hat values - they need to be close to 1 if convergence has been achieved
plot(model_run)
print(model_run)
traceplot(model_run)

# Create a plot of the posterior mean regression line
post <- print(model_run)
alpha_mean <- post$mean$alpha
beta_mean <- post$mean$beta

plot(x, y)
lines(x, inv.logit(alpha_mean + beta_mean * x), col = "red")
lines(x, inv.logit(alpha + beta * x), col = "blue")
legend("topleft",
       legend = c("Truth", "Posterior mean"),
       lty = 1,
       col = c("blue", "red")
)
# Blue and red lines should be pretty close

# Real example ------------------------------------------------------------

# Load in
library(datasets)
head(attenu)

# Set up the data
acc <- with(attenu, list(
  y = attenu$accel,
  x = attenu$dist,
  T = nrow(attenu)
))
# Plot
plot(attenu$dist, attenu$accel)

# Set up jags model
jags_model <- jags(acc,
                   parameters.to.save = model_parameters,
                   model.file = textConnection(model_code),
                   n.chains = 4,
                   n.iter = 1000,
                   n.burnin = 200,
                   n.thin = 2
)
# Plot the jags output
print(jags_model)
traceplot(jags_model)

# Plot of posterior line
post <- print(jags_model)
alpha_mean <- post$mean$alpha
beta_mean <- post$mean$beta

