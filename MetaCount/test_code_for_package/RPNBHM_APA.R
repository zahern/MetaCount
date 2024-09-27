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

Ndraws=50      # set number of draws 
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
dataH = as.matrix(data.frame(dataset$HSP))

# separating the variables with random parameters 
dataR = as.matrix(data.frame(log(dataset$AADT),dataset$Lwidth))

dataR2=NULL
Dvar2 = NULL
for(i in 1:Ndraws){
  dataR2=rbind(dataR2,dataR)
  Dvar2 = c(Dvar2,DVar)
}

draws1 = draws1[,1:dimensions]
# separating the variables with fixed parameters 
dataF =  as.matrix(data.frame(1,log(dataset$Length)))

#Seperating the variables with hetrogeneity in the means (for each random parameter, you need one vector of variables)
dataH1 = as.matrix(data.frame(dataset$HSP,dataset$MCV))
dataH2 = as.matrix(data.frame(dataset$LSP,dataset$AADT,dataset$Length))

# Likelihood function
LL <- function(params){  
  disp <- params[1] # Dispersion Parameter (comment out and adjust params[] for the Poisson model)
  Fbeta <- params[2:3] # Fixed parameters in the mean Function
  HMbeta1 <-params[4:5] # Parameter for hetrogeneity in the means of the 1st random parameter
  HMbeta2 <-params[6:8] # Parameter for hetrogeneity in the means of the 2nd random parameter
  SDRbeta <- params[9:10]  # Std of Random parameters in the mean function
  
  # vector of indipendent variables with fixed parameters
  offset = rep.int(dataF%*%as.matrix(Fbeta,ncol=1),Ndraws)
  
  # heterogeneity in means
  MRbeta1 = rep.int(dataH1%*%as.matrix(HMbeta1,ncol=1),Ndraws)
  MRbeta2 = rep.int(dataH2%*%as.matrix(HMbeta2,ncol=1),Ndraws)
  
  MRbeta = matrix(cbind(MRbeta1,MRbeta2),ncol = 2)
  
  # simulating random parameters from their means and standard deviation
  beta = t( t(draws1)*SDRbeta ) + MRbeta
  
  # constructing the mean function
  mu <- exp(offset+rowSums(dataR2*beta))
  
  # simulated maximum loglikelihood for negative binomial distribution
  loglik <-  sum(log(rowMeans(matrix(dnbinom(Dvar2,size=disp,mu=mu,log = F), ncol = Ndraws))))
  
  return(loglik)
}

# initial values for optimization
init <- c(0.5,#dispersion parameter(remove for the Poisson model)
          -20,2.6,#fixed parameters
          0.17,-0.14,#heterogeneity in the mean of random parameter1
          0.1,0.1,0.1,#heterogeneity in the mean of random parameter2
          0.05,0.08,#standard deviation of random parameters
          0.5) #hetrogeneity in the means

# optimization (maximization of likelihood function)
fit1 <- maxLik(LL,start=init,method="BFGS")

summary(fit1)

