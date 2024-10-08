
library(remotes)

#library(finiteMix)


R_LINDL <- function(n, beta){
  mean = array(unlist(rlindley(n, beta, FALSE)[1]))
  return(mean)
}

R_GAMMA <- function(n, a, b){
  rgamma(n = n, shape = a)
}






