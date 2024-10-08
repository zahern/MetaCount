rm(list = ls())
require(stats4)
require(maxLik)
require(randtoolbox)

# reading and storing data in a dataframe
dataset <- read.csv(file.choose(),header=T)
#Sample Size
N <- nrow(dataset) 
#Dependent variable (crash counts in this example); change the variable as required
DVar <- dataset$Y 







## data preparation
# separating the variables with fixed parameters 
dataF =  as.matrix(data.frame(1))
# separating the variables with random parameters 
disp <- 0.05 # Dispersion Parameter (comment out and adjust params[] for the Poisson model)
Fbeta <- -5.56 # Fixed parameters in the mean Function
#MRbeta <- params[4:5]  # Mean of Random parameters in the mean function
#SDRbeta <- params[6:7]  # Std of Random parameters in the mean function

# vector of indipendent variables with fixed parameters
offset = dataF%*%as.matrix(Fbeta,ncol=1)
# simulating random parameters from their means and standard deviation
#beta = t( t(draws1)*SDRbeta + MRbeta )
# constructing the mean function
mu <- exp(offset)
log(rowMeans(matrix(dnbinom(DVar,size=disp,mu=mu,log = F), ncol = 1)))
sudnbinom(DVar,size=disp,mu=mu,log = F)
# Likelihood function
LL <- function(params){  
  disp <- params[1] # Dispersion Parameter (comment out and adjust params[] for the Poisson model)
  Fbeta <- params[2] # Fixed parameters in the mean Function
  #MRbeta <- params[4:5]  # Mean of Random parameters in the mean function
  #SDRbeta <- params[6:7]  # Std of Random parameters in the mean function
  
  # vector of indipendent variables with fixed parameters
  offset = dataF%*%as.matrix(Fbeta,ncol=1)
  # simulating random parameters from their means and standard deviation
  #beta = t( t(draws1)*SDRbeta + MRbeta )
  # constructing the mean function
  mu <- exp(offset)
  # simulated maximum loglikelihood for negative binomial distribution
  loglik <-  sum(log(rowMeans(matrix(dnbinom(DVar,size=disp,mu=mu,log = F), ncol = 1))))
  # simulated maximum loglikelihood for Poisson distribution
  # loglik <-  sum(log(rowMeans(matrix(dpois(Dvar2,lambda=mu,log = F), ncol = Ndraws))))
  
  return(loglik)
}

# initial values for optimization
init <- c(1,#dispersion parameter(remove for the Poisson model)
          1)#standard deviation of random parameters

# optimization (maximization of likelihood function)
fit1 <- maxLik(LL,start=init,method="BFGS")

summary(fit1)

# Predictions, Residuals and Measures of Fit 

params <- fit1$estimate
Fbeta <- params[2:3] # Fixed parameters in Mu Function
MRbeta <- params[4:5]  # Mean of Random parameters in Mu function

offset = as.vector(dataF%*%as.matrix(Fbeta,ncol=1))
mu <- exp(offset+log(dataset$AADT)*MRbeta[1]+dataset$LWIDTH*MRbeta[2])

# fitted values
muFit <- mu
# residuals
RESIDS <- DVar-muFit
# absolute residuals
ABSRES <- abs(RESIDS)
# squared residuals
SQRES <- (RESIDS)^2
# number of estimated parameters in the model
P <- NROW(params)
# mean absolute deviance
MAD <- sum(ABSRES)/N
# mean squared predictive error
MSPE <- sum(SQRES)/N
# goodness of fit
GOF <- data.frame(MAD=MAD,MSPE=MSPE) 
  
