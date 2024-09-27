
library(remotes)
#library(CMP)

library(COUNT)
#library(saeHB.gpois)

library(VGAM)
library(RNGforGPD)
library(HMMpa)
library(gamlss)
library(gamlss.dist)


CONWAY <- function(n, lam, nu){
  mean = rcomp(n = n, lambda = lam, nu=1)
  return(mean)
}

GENPON <- function(n, lam, nu){
   mean = rgenpois2(n, lam, nu)
   
   return(mean)
   }
 


GENUNI <- function(lam, nu){
   #mean = GenUniGpois(lam, nu, 10, FALSE, "Inversion")
   mean = rgenpois1(1, lam, nu)
   return(mean)
}

GPON <- function(mu, sigma){
  fsis <- rGPO(mu, sigma)
  
  return(fsis)
}

GPOS <- function(mu){
  fsi =rpois(mu)
  return(fsi)
}


