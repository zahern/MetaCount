rm(list = ls())
require(stats4)
require(maxLik)
require(randtoolbox)

# reading and storing data in a dataframe
dataset <- read.csv(file.choose(),header=T)
#Sample Size
N <- nrow(dataset) 
#Dependent variable (crash counts in this example); change the variable as required
DVar <- dataset$Headon

# Halton Draws 
preparedraws=function()
{
  d=1
  while(d<(length(normaldraws)+1))
  {
    draws1[,normaldraws[d]]<<- qnorm(draws1[,normaldraws[d]])
    d=d+1
  }
}

Ndraws=200      # set number of draws 
dimensions=2    # define number of random parameters in the model

# generate draws (using Halton)
draws1=as.matrix(halton(Ndraws*N,dimensions))

# assign names to individual sets of draws - need one entry per dimension
colnames(draws1)=c("HRbeta1","HRbeta2")
# define whether any draws should be transformed to Normals, which is also needed for e.g. lognormals (leave empty if not)
normaldraws=c("HRbeta1","HRbeta2")

# preparing draws for estimation - this may take a while
preparedraws()

#I have commented this out because no panel
# fixing parameters across grouped observations i.e. grouped random parameters
# Do not use if there is no panel
#block = length(unique(dataset[,'ID']))
#ngroup = length(unique(dataset[,'Group']))
#for (i in 1:Ndraws){
#  tempInd = ((i-1)*block*ngroup) + (1:block)
 # for (ii in 2:ngroup){
 #   draws1[tempInd+(ii-1)*block,] = draws1[tempInd,]
 # }
#}

## data preparation
# separating the variables with fixed parameters 
dataF =  as.matrix(data.frame(1,log(dataset$Length)))

#Seperating the variables with hetrogeneity in the means
dataH = as.matrix(data.frame(dataset$S_Curve))




# separating the variables with random parameters 
dataR = as.matrix(data.frame(log(dataset$AADT),dataset$Lwidth))

dataR2=NULL
Dvar2 = NULL
for(i in 1:Ndraws){
  dataR2=rbind(dataR2,dataR)
  Dvar2 = c(Dvar2,DVar)
}

draws1 = draws1[,1:dimensions]







# Likelihood function
LL <- function(params){  
  
  Fbeta <- params[1:2] # Fixed parameters in the mean Function
  HMbeta <-params[3] #Parameter for hetrogeneity in the means
  MRbeta <- params[4:5]  # Mean of Random parameters in the mean function
  SDRbeta <- params[6:7]  # Std of Random parameters in the mean function
  
  #disp <- params[1] # Dispersion Parameter (comment out and adjust params[] for the Poisson model)
  #hetrogeneity in the means
  hm <- dataH%*%(as.matrix(Hbeta,ncol=1))
  new_hm <- matrix(0, nrow = nrow(hm), ncol = length(Fbeta))
  for (i in 1:length(Fbeta)){

    new_hm[,i] <- hm +Fbeta[i]
  }
  # vector of indipendent variables with fixed parameters
  offset = rep.int(rowSums(new_hm*dataF),Ndraws)
  # simulating random parameters from their means and standard deviation
  beta = t( t(draws1)*SDRbeta + MRbeta )
  # constructing the mean function
  mu <- exp(offset+rowSums(dataR2*beta))
  # simulated maximum loglikelihood for negative binomial distribution
  #loglik <-  sum(log(rowMeans(matrix(dnbinom(Dvar2,size=disp,mu=mu,log = F), ncol = Ndraws))))
  # simulated maximum loglikelihood for Poisson distribution
   loglik <-  sum(log(rowMeans(matrix(dpois(Dvar2,lambda=mu,log = F), ncol = Ndraws))))
  
  return(loglik)
}

# initial values for optimization
init <- c(-8,.5,#fixed parameters
          0.5, #hetrogeneity in the means
          0.69,-0.26,#mean of random parameters
          0.05,0.08)#standard deviation of random parameters
          

# optimization (maximization of likelihood function)
fit1 <- maxLik(LL,start=init,method="BFGS")

summary(fit1)

# Predictions, Residuals and Measures of Fit 

params <- fit1$estimate
Fbeta <- params[2:3] # Fixed parameters in Mu Function
MRbeta <- params[4:5]  # Mean of Random parameters in Mu function

offset = as.vector(dataF%*%as.matrix(Fbeta,ncol=1))
mu <- exp(offset+log(dataset$AADT)*MRbeta[1]+dataset$Lwidth*MRbeta[2])

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
