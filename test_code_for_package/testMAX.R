library(maxLik)

#simulate data



LLFUN <- function(loglik, start){
  fit1 <- maxLik(logLik = loglik, start = start, method = 'BFGS')
  summary(fit1)
}




