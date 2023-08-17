library(DDD)
library(MLmetrics)
library(dplyr)
library(ape)
library(diversitree)
library(RPANDA)
library(latex2exp)
library(castor)
library(phangorn)
library(svMisc)
library(torch)
library(igraph)
library(scales)
library(extraDistr)

source("R/infer-general-functions.R")#Contains functions for generating the phylogenetic trees ,plotting, computing rates.
source("R/convert-phylo-to-sumstat.R")#contains the functions to compute the summary statistics of a phylogenetic tree(tips,depth)
source("R/convert-phylo-to-cblv.R") #contains the function to encode a phylogenetic tree with the "Compact Bijective Ladderized
source("R/new_funcs.R")



n_trees<-10000
phylo_nameddd<-"phylogeny-DDD2-nt-10000-la0-0-50-mu-0-50-k-20-400-age-1-ddmod-10.rds"
brts_nameddd<-"brts-DDD2-nt-10000-la0-0-50-mu-0-50-k-20-400-age-1-ddmod-10.rds"

  
  
phylo <- readRDS( paste("data_DDD/",phylo_nameddd,sep=""))

brts <-readRDS( paste("data_DDD/",brts_nameddd,sep=""))




MLE_stimates <- list()


start_time <- Sys.time()
for( i in 1:length(brts)){
  mle=dd_ML(brts = brts[[i]], idparsopt = c(1,2,3),cond = 1,  optimmethod = 'simplex',ddmodel=1)
  MLE_stimates<- append(MLE_stimates, list(mle)) 
  
  progress(length(brts), n_test, progress.bar = TRUE, # print
           init = (length(brts_test)==1))
}
end_time <- Sys.time()


result <- list("time"= 0 , "units"= "")

time<- as.numeric(end_time - start_time)
unit<-units(end_time - start_time)

result$time<-time
result$units<-unit

write.csv(result, file = "data_DDD/MLE_time.csv", row.names = FALSE)

saveRDS(MLE_stimates, paste("data_DDD/MLE-DDD2-nt-10000-la0-0-50-mu-0-50-k-20-400-age-1-ddmod-10.rds", sep=""))

