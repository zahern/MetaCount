# Negative-Binomial-Lindly distribution
# rnbl(n,r,theta) - random numbers from NBL
# dnbl(x,r,theta) - NBL pmf at x
# pnbl(q,r,theta) - NBL cdf at x
# qnbl(p,r,theta) - NBL inverse at p
# Soma S Dhavala, 11/5/2013
# somasd@gmail.com
# GNU Public License, free to use/copy with/without attribution


# nbl random numbers
rnbl <- function(n,r,theta)
{
  
  lambda <- ifelse(runif(n) < theta/(1 + theta), rexp(n, theta), 
                   rgamma(n,shape = 2, scale = 1/theta))
  p <- exp(-lambda)
  x <- rnbinom(n,r,prob=p)
}


# pmf of a vector based on numerical integration
dnbl <- function(x,r,theta)
{
  
  nbl.integrand <- function(x,k,r,theta)
  {
    nlogp <- -log(x)
    tmp1 <- dnbinom(k,r,x,log=TRUE)
    tmp2 <- -theta*nlogp
    tmp3 <- 2*log(theta)-log(1+theta)
    tmp4 <- log(1+nlogp)
    y <- exp(nlogp+tmp1+tmp2+tmp3+tmp4)
    return(y)
    
  }
  nbl.pmf <- function(x,r,theta)
  {
    return(integrate(nbl.integrand,lower=0,upper=1,k=x,r=r,theta=theta)$value)
  }
  p <- sapply(x,nbl.pmf,r=r,theta=theta)
  return(p)
}

# cdf of a vector based on numerical integration
pnbl <- function(x,r,theta)
{
  
  nbl.integrand <- function(x,k,r,theta)
  {
    nlogp <- -log(x)
    tmp1 <- pnbinom(k,r,x,log.p=TRUE)
    tmp2 <- -theta*nlogp
    tmp3 <- 2*log(theta)-log(1+theta)
    tmp4 <- log(1+nlogp)
    y <- exp(nlogp+tmp1+tmp2+tmp3+tmp4)
    return(y)
    
  }
  nbl.cdf <- function(x,r,theta)
  {
    return(integrate(nbl.integrand,lower=0,upper=1,k=x,r=r,theta=theta)$value)
  }
  p <- sapply(x,nbl.cdf,r=r,theta=theta)
  return(p)
}

# inverse-cdf but based on monte-carlo estimates
qnbl <- function(p,r,theta,nboot=10000)
{
  x <- rnbl(nboot,r,theta)
  q <- quantile(x,probs=p)
  return(q)
}

# example
theta <- 100
r <- 20
n = 1000

dnbl(c(0,1,2),r,theta)
pnbl(c(0,1,2),r,theta)
qnbl(c(0.1,0.5,0.88,0.99),r,theta)
