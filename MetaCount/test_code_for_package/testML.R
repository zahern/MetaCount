
library(mlbench)
library(randomForest)
library(forestFloor)
library(AUC)
library(dplyr)
library(caret)
library(randomForest)
library(reticulate)
library(tree)

#simulate data









RF_plot_corr <- function(X, Y) {
  print(dim(X))
  print('why')
  #long format the X
  # Create all two-way interaction terms
  colnames <- colnames(X)
  for (i in 1:(ncol(X) - 1)) {
    for (j in (i + 1):ncol(X)) {
      X[,paste(colnames[i], colnames[j], sep = "_")] <- X[,i] * X[,j]
    }
  }
  print('does this expand')
  print(colnames(X))
  
  cmat <- cor(X)
  diag(cmat) <- 0
 
  highcorr <- findCorrelation(cmat, cutoff = .8)
  negated_array <- sapply(highcorr, FUN = function(x) !x)
  print(highcorr)
  print(negated_array)
  X = X[, negated_array]
  print(dim(X))
  print('cool')
  
  rfo=randomForest(X,Y[,1],keep.inbag = TRUE, sampsize = 200, ntree=270)
  
  #compute topology
  ff = forestFloor(rfo,X)
  Col = fcol(ff,3,orderByImportance=TRUE)
  if (dim(X)[2] > 15){
    plot(ff,col=Col,compute_GOF=TRUE, mfrow = c(4, ceiling(dim(X)[2]/4)))
  }
  else{
    #plot(ff, orderByImportance=TRUE)
    
    #jpeg('rplot.jpg')
    plot(ff,col=Col,compute_GOF=TRUE)
  }
  tree.model2 = tree(Y[,1]~X)
  plot(tree.model2)
  
  
}

RF_plot <- function(X, Y) {
    print(dim(X))
    print(dim(Y))
    
    rfo=randomForest(X,Y[,1],keep.inbag = TRUE, sampsize = 200, ntree=270)

    #compute topology
    ff = forestFloor(rfo,X)
    Col = fcol(ff,3,orderByImportance=TRUE)
    if (dim(X)[2] > 15){
      plot(ff,col=Col,compute_GOF=TRUE, mfrow = c(4, ceiling(dim(X)[2]/4)))
    }
    else{
    #plot(ff, orderByImportance=TRUE)
    
    #jpeg('rplot.jpg')
    plot(ff,col=Col,compute_GOF=TRUE)
    }
    
  
  
}

#print forestFloor
#print(ff) 

#plot partial functions of most important variables first
#plot(ff) 

#Non interacting functions are well displayed, whereas X3 and X4 are not
#by applying different colourgradient, interactions reveal themself 



