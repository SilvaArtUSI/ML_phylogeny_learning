


library(DDD)
library(ape)
library(diversitree)
library(RPANDA)
library(latex2exp)
library(castor)
library(phangorn)
library(svMisc)
library(igraph)
library(scales)
library(emphasis)
#library(caret)

source("R/infer-general-functions.R")#Contains functions for generating the phylogenetic trees ,plotting, computing rates.
source("R/convert-phylo-to-sumstat.R")#contains the functions to compute the summary statistics of a phylogenetic tree(tips,depth)
source("R/convert-phylo-to-cblv.R") #contains the function to encode a phylogenetic tree with the "Compact Bijective Ladderized
source("R/convert-phylo-to-graph.R") #contains the function to encode a phylogenetic tree with the "Compact Bijective Ladderized
source("R/new_funcs.R")




path_dir = "/Users/oasc/Documents/Thesis/ML_phylogeny_learning"
setwd(path_dir)


source("R/infer-general-functions.R")#Contains functions for generating the phylogenetic trees ,plotting, computing rates.
source("R/convert-phylo-to-sumstat.R")#contains the functions to compute the summary statistics of a phylogenetic tree(tips,depth)
source("R/convert-phylo-to-cblv.R") #contains the function to encode a phylogenetic tree with the "Compact Bijective Ladderized
source("R/convert-phylo-to-graph.R") #contains the function to encode a phylogenetic tree with the "Compact Bijective Ladderized
source("R/new_funcs.R")




phylo_crbd <- readRDS( paste("data_clas/phylogeny-crbd10000ld-01-1-e-0-9.rds", sep=""))
phylo_bisse <- readRDS( paste("data_clas/phylogeny-bisse-10000ld-.01-1.0-q-.01-.1.rds", sep=""))




phylo_crbdrescaled <- lapply(phylo_crbd, rescale_tree)
phylo_bisserescaled<-lapply(phylo_bisse, rescale_tree)

saveRDS(phylo_crbdrescaled, paste("data_clas/phylogeny-reescrbd",10000,"ld-01-1-e-0-9.rds", sep=""))
saveRDS(phylo_bisserescaled, paste("data_clas/phylogeny-reesbisse-10000ld-.01-1.0-q-.01-.1.rds", sep=""))

rm(phylo_bisse)
rm(phylo_crbd)

true_crbd <- readRDS( paste("data_clas/true-param-crbd10000ld-01-1-e-0-9.rds", sep=""))
true_bisse <- readRDS( paste("data_clas/true-param-bisse-10000ld-.01-1.0-q-.01-.1.rds", sep=""))


start_time<- Sys.time()
sumstat_crbd<-generateSumStatFromPhylo(phylo_crbdrescaled, true_crbd) 
end_time <- Sys.time()
t1<-end_time - start_time
print(t1)
start_time<- Sys.time()
sumstat_bisse<-generateSumStatFromPhylo(phylo_bisserescaled, true_bisse) 
end_time <- Sys.time()
t2<-end_time - start_time
print(t2)


saveRDS(sumstat_crbd, paste("data_clas/phylogeny-reescrbd-",10000,"ld-01-1-e-0-9-sumstat.rds", sep=""))
saveRDS(sumstat_bisse, paste("data_clas/phylogeny-reesbisse-",10000,"ld-01-1-e-0-9-sumstat.rds", sep=""))


max_nodes=1200
start_time<- Sys.time()
cblv_crbd_res <- generate_encoding_DDD(phylo_crbdrescaled, tree_size = max_nodes)
end_time <- Sys.time()
t3<-end_time - start_time
print(t3)

start_time<- Sys.time()
cblv_bisse_res <- generate_encoding_DDD(phylo_bisserescaled, tree_size = max_nodes)
end_time <- Sys.time()
t4<-end_time - start_time
print(t4)


saveRDS(cblv_crbd_res, paste("data_clas/phylogeny-reescrbd-",10000,"ld-01-1-e-0-9-cblv.rds", sep=""))
saveRDS(cblv_bisse_res, paste("data_clas/phylogeny-reesbisse-",10000,"ld-01-1-e-0-9-cblv.rds", sep=""))


## encoding for grpah

phylo_ddd <- readRDS( paste("data_clas/phylogeny-DDD2-nt-10000-la0-0-50-mu-0-50-k-20-400-age-1-ddmod-10.rds", sep=""))
phylo_pld <- readRDS( paste("data_clas/phylogeny-pld-nt-10000-la0-0-50-mu-0-50-k-20-400-age-1-ddmod-10.rds", sep=""))


n_trees=10000
phylo_re<-c(phylo_crbdrescaled,phylo_bisserescaled,phylo_ddd,phylo_pld)
min_nodes<-getmin_nodes(phylo_re)


## Generate LTT  fro CNNLTT and LSTM



true_crbd   <-   list("crbd"  = rep(2, n_trees), 
                      "bisse" = rep(1, n_trees), 
                      "ddd"  = rep(1, n_trees),
                      "pld"  = rep(1, n_trees))


true_bisse   <-   list("crbd"  = rep(1, n_trees), 
                       "bisse" = rep(2, n_trees), 
                       "ddd"  = rep(1, n_trees),
                       "pld"  = rep(1, n_trees))


true_ddd   <-   list("crbd"  = rep(1, n_trees), 
                     "bisse" = rep(1, n_trees), 
                     "ddd"  = rep(2, n_trees),
                     "pld"  = rep(1, n_trees))

true_pld   <-   list("crbd"  = rep(1, n_trees), 
                     "bisse" = rep(1, n_trees), 
                     "ddd"  = rep(1, n_trees),
                     "pld"  = rep(2, n_trees))

true_names<-names(true_crbd)


true <- lapply(1:4, function(i) c(true_crbd[[i]], true_bisse[[i]], true_ddd[[i]], true_pld[[i]]))
names(true)<-true_names


saveRDS(true,  paste("data_clas/phylo-true-clas-all2.rds", sep=""))


print("Generating LTT")
start_time<- Sys.time()
df.ltt_all<-generate_ltt_dataframe(phylo_re, max_nodes, true)$ltt
end_time <- Sys.time()
print(end_time - start_time)
print("LTT finished Saving")
saveRDS(df.ltt_all,  paste("data_clas/phyloreescaled-all-dfltt.rds", sep=""))




print("Generating graph info")
start_time<- Sys.time()
graph_all <- generate_phylogeny_graph(phylo_re)
end_time <- Sys.time()

saveRDS(graph_all, paste("data_clas/phylogeny_re-all-graph.rds",sep=""))
print("Graph finished")

#


