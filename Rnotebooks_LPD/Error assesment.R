
#Loading phylo


library(torch)
library(luz)
library(vioplot)
library(ggplot2)
library(dplyr)
library(hrbrthemes)
library(viridis)
path_dir = "/Users/oasc/Documents/Thesis/ML_phylogeny_learning"


source( "R/neural-network-functions.R")
source( "R/infer-general-functions.R")
source( "R/phylo-inference-ml.R")
source( "R/new_funcs.R")
source( "R/phylo_eval.R")


device<-"cpu"
n_trees<-10000



path_pld="data_PLD/"

phylo_namePLD<-"phylogeny-pld-nt-10000-la0-0-50-mu-0-50-k-20-400-age-1-ddmod-10.rds"
true_namePLD<-"true-param-pld-nt-10000-la0-0-50-mu-0-50-k-20-400-age-1-ddmod-10.rds"

phylo <- readRDS(paste( "data_PLD/",phylo_namePLD,sep =""))
true<-readRDS(paste( "data_PLD/",true_namePLD,sep=""))


cblv<- readRDS("data_PLD/phylogeny-PLD-10000-cblv.rds")
df.ltt<-readRDS("data_PLD/phylo_DDD_dfltt.rds")

lambda0_pld <- c(0, 50) # speciation rate
beta_n <-c(-2.5,0) 
beta_p <-c(-2.5,2.5) 

#true[[4]]<-true[[4]]/100 #adjusting parameters

#true<-true[c(-4)] # Removing crown age from variables


true[[1]]<-min_max_normalize(true[[1]],rang=lambda0_pld)
true[[2]]<-min_max_normalize(true[[2]],rang=lambda0_pld)
true[[3]]<-min_max_normalize(true[[3]],rang=beta_n)
true[[4]]<-min_max_normalize(true[[4]],rang=beta_p)







max_nodes_rounded <- 1200 #this was encoded with all the phylogenies



set.seed(113)
# Define size of datasets.
total_data_points <- n_trees
subset_size <- 10000  # Specify the size of the subset

n_train    <- floor(subset_size * .9)
n_valid    <- floor(subset_size * .05)
n_test     <- subset_size - n_train - n_valid
batch_size <- batch_size <- min(subset_size*.01, 10)
subset_indices <- sample(1:ncol(cblv), subset_size)
test_indices  <- subset_indices[(n_train + n_valid + 1):subset_size]

# Pick the phylogenies randomly.

ds_cnn <- convert_encode_to_dataset(cblv, true)
test_ds_cnn  <- ds_cnn(cblv[1:nrow(cblv), test_indices], extract_elements(true, test_indices))
test_dl_cnn  <- test_ds_cnn  %>% dataloader(batch_size=1, shuffle=FALSE)


####Realted to cblv encoding

ds.ltt <- convert_ltt_dataframe_to_dataset(df.ltt, true, "cnn-ltt")
test_ds_ltt  <- ds.ltt(df.ltt[, test_indices] , extract_elements(true, test_indices))
test_dl_ltt  <- test_ds_ltt  %>% dataloader(batch_size=1,          shuffle=FALSE)


source("PLD_Notebooks/PLD_NN_source.R")

rm(nn.pred_cnn)
rm(nn.pred_ltt)

n_out<-4
nn.pred_ltt <- vector(mode = "list", length = n_out)
names(nn.pred_ltt) <- names(c("lambda0", "mu","beta_n","beta_p" ))

nn.pred_cnn <- vector(mode = "list", length = n_out)
names(nn.pred_cnn ) <- names(c("lambda0","mu","beta_n","beta_p"  ))



####Computing for CNN
cnn$eval()
p_dropout <- 0
# Compute predictions 
coro::loop(for (b in test_dl_cnn) {
  out <- cnn(b$x$to(device = device))
  p <- as.numeric(out$to(device = "cpu")) # move the tensor to CPU 
  for (i in 1:n_out){nn.pred_cnn[[i]] <- c(nn.pred_cnn[[i]], p[i])}
})

####Computing for LTT
cnn_ltt$eval()
p_dropout <- 0
# Compute predictions 
coro::loop(for (b in test_dl_ltt) {
  #if (model_type == "crbd"){b$x <- b$x$unsqueeze(2)}
  out <- cnn_ltt(b$x$to(device = device))
  pred <- as.numeric(out$to(device = "cpu")) # move the tensor to CPU 
  #true <- as.numeric(b$y)
  for (i in 1:n_out){nn.pred_ltt[[i]] <- c(nn.pred_ltt[[i]], pred[i])}
})



nn.pred_cnn[[1]]<-min_max_unnormalize(nn.pred_cnn[[1]],rang=lambda0_pld)
nn.pred_cnn[[2]]<-min_max_unnormalize(nn.pred_cnn[[2]],rang=lambda0_pld)
nn.pred_cnn[[3]]<-min_max_unnormalize(nn.pred_cnn[[3]],rang=beta_n)
nn.pred_cnn[[4]]<-min_max_unnormalize(nn.pred_cnn[[4]],rang=beta_p)



nn.pred_ltt[[1]]<-min_max_unnormalize(nn.pred_ltt[[1]],rang=lambda0_pld)
nn.pred_ltt[[2]]<-min_max_unnormalize(nn.pred_ltt[[2]],rang=lambda0_pld)
nn.pred_ltt[[3]]<-min_max_unnormalize(nn.pred_ltt[[3]],rang=beta_n)
nn.pred_ltt[[4]]<-min_max_unnormalize(nn.pred_ltt[[4]],rang=beta_p)




dpi=300
factor=floor(dpi/150)

true<-readRDS(paste( "data_PLD/",true_namePLD,sep=""))
true$beta_n <- (true$mu-true$lambda0)/true$K
true<-true[c(1,2,5,4)]
#Plots
###CNN
png("data_PLD/plots/cnn_estimated_500.png", width = 480*3*factor, height = 480*factor,res=dpi)
par(mfrow = c(1, 4))
plot(true[[1]][test_indices], nn.pred_cnn[[1]], main = "lambda0", xlab = "True", ylab = "Predicted")
abline(0, 1,col="red")
plot(true[[2]][test_indices], nn.pred_cnn[[2]], main = "mu", xlab = "True", ylab = "Predicted")
abline(0, 1,col="red")
plot(true[[3]][test_indices] , nn.pred_cnn[[3]] , main = "K", xlab = "True", ylab = "Predicted")
abline(0, 1,col="red")
plot(true[[4]][test_indices]  ,  nn.pred_cnn[[4]] , main = "Beta", xlab = "True", ylab ="Predicted")
abline(0, 1,col="red")
dev.off()

#Plots
###CNNLTT
png("data_PLD/plots/cnnltt_estimated_500.png", width = 480*3*factor, height = 480*factor,res=dpi)
par(mfrow = c(1, 4))
plot(true[[1]][test_indices], nn.pred_ltt[[1]], main = "lambda0", xlab = "True", ylab = "Predicted")
abline(0, 1,col="red")
plot(true[[2]][test_indices], nn.pred_ltt[[2]], main = "mu", xlab = "True", ylab = "Predicted")
abline(0, 1,col="red")
plot(true[[3]][test_indices] , nn.pred_ltt[[3]] , main = "K", xlab = "True", ylab = "Predicted")
abline(0, 1,col="red")
plot(true[[4]][test_indices]  ,  nn.pred_ltt[[4]] , main = "Beta", xlab = "True", ylab ="Predicted")
abline(0, 1,col="red")
dev.off()





error_cnn <- vector(mode = "list", length = 4)
error_cnnltt <- vector(mode = "list", length = 4)


for (i in 1:4){
  error_cnnltt[[i]] <- true[[i]]-nn.pred_ltt[[i]]
  error_cnn[[i]] <- true[[i]]-nn.pred_cnn[[i]]
  
}



#column_names <- c("lambda0", "mu","K" ,"Beta_N")
column_names <- c("lambda0", "mu", "beta n" ,"beta p")
# Convert list to dataframe with custom column names

df.error_cnnltt <- data.frame(error_cnnltt,check.names = FALSE)
colnames(df.error_cnnltt) = column_names

df.error_cnn<- data.frame(error_cnn,check.names = FALSE)
colnames(df.error_cnn) = column_names


# Print the resulting dataframe
#print(my_dataframe)

#df.error_nn$method = "NN"
#df.error_mle$method = "MLE"


#df.error=  rbind(df.error_nn[rmoutliers,],df.error_mle[rmoutliers,])


#ggplot(df.error) + geom_point(aes(x=lambda0,y=mu,colour=method))+theme_minimal()






# Melt the data frame for easier plotting
melted_data_cnnltt <- reshape2::melt(df.error_cnnltt)
melted_data_cnn <- reshape2::melt(df.error_cnn)

# Create a custom color palette
my_palette <- c("#FFA500", "#008080", "#4DAF4A", "#984EA3")

#"#E41A1C", "#377EB8", "#4DAF4A", "#984EA3")
#Orange: #FFA500
#  Pink: #FFC0CB
#  Teal: #008080
#  Brown: #A52A2A

# Create a violin plot with facets for each column
cnnltt_vio<-ggplot(melted_data_cnnltt, aes(x = variable, y = value, fill = variable)) +
  geom_violin() +
  geom_boxplot(width = 0.2, color = "black", fill = "white") +
  labs(x = "", y = "Error") +
  facet_wrap(~ variable, scales = "free",nrow = 1, ncol = 5) +
  scale_fill_manual(values = my_palette) +  # Apply custom color palette
  theme_minimal() +
  guides(fill = "none") 


cnn_vio<-ggplot(melted_data_cnn, aes(x = variable, y = value, fill = variable)) +
  geom_violin() +
  geom_boxplot(width = 0.2, color = "black", fill = "white") +
  labs(x = "", y = "Error") +
  facet_wrap(~ variable, scales = "free",nrow = 1, ncol = 5) +
  scale_fill_manual(values = my_palette) +  # Apply custom color palette
  theme_minimal() +
  guides(fill = "none") 


# Save the plot to a PNG file

png("data_PLD/plots/violin_plotcnnltt.png", width = 800, height = 300,res=125)
print(cnnltt_vio)
dev.off()

png("data_PLD/plots/violin_plotcnn.png", width = 800, height = 300,res=125)
print(cnn_vio)
dev.off()





true_wo_k<-extract_elements(true, test_indices)
nn.pred_cnn_wo_k<-nn.pred_cnn
nn.pred_ltt_wo_k<-nn.pred_ltt


mse_cnn <- list()
mse_cnnltt <- list()

for (i in 1:length(true_wo_k))
{
  s_mse_cnn<- sqrt(mean((true_wo_k[[i]]-nn.pred_cnn_wo_k[[i]])^2))/abs(mean(true_wo_k[[i]]))
  mse_cnn<-c(mse_cnn, s_mse_cnn)
  
  
  s_mse_cnnltt<- sqrt(mean((true_wo_k[[i]]-nn.pred_ltt_wo_k[[i]])^2))/abs(mean(true_wo_k[[i]]))
  mse_cnnltt<-c(mse_cnnltt, s_mse_cnnltt)

  
  
}

mse_vf<-rbind(mse_cnn,mse_cnnltt)

colnames(mse_vf)<-column_names

write.csv(mse_vf, file = "data_PLD/results/MSE_ALL_WOK_16_norm.csv", row.names = TRUE)




#Plots
###CNN
png("data_PLD/plots/cnn_estimated_100_wok.png", width = 480*3*factor, height = 480*factor,res=dpi)
par(mfrow = c(1, 3))
plot(true[[1]], nn.pred_cnn[[1]], main = "lambda0", xlab = "True", ylab = "Predicted")
abline(0, 1,col="red")
plot(true[[2]], nn.pred_cnn[[2]], main = "mu", xlab = "True", ylab = "Predicted")
abline(0, 1,col="red")
plot(true[[4]]  ,  nn.pred_cnn[[4]] , main = "Beta", xlab = "True", ylab ="Predicted",xlim=c(-.6,0),ylim=c(-.6,0))
abline(0, 1,col="red")
dev.off()

#Plots
###CNNLTT
png("data_PLD/plots/cnnltt_estimated_100_wok.png", width = 480*3*factor, height = 480*factor,res=dpi)
par(mfrow = c(1, 3))
plot(true[[1]], nn.pred_ltt[[1]], main = "lambda0", xlab = "True", ylab = "Predicted")
abline(0, 1,col="red")
plot(true[[2]], nn.pred_ltt[[2]], main = "mu", xlab = "True", ylab = "Predicted")
abline(0, 1,col="red")
plot(true[[4]]  ,  nn.pred_ltt[[4]] , main = "Beta", xlab = "True", ylab ="Predicted",xlim=c(-.6,0),ylim=c(-.6,0))
abline(0, 1,col="red")
dev.off()

#Plots
###MLE
png("data_PLD/plots/mle_estimated_100_wok.png", width = 480*3*factor, height = 480*factor,res=dpi)
par(mfrow = c(1, 3))
plot(true[[1]], mle[[1]], main = "lambda0", xlab = "True", ylab = "Predicted",xlim=c(0,50),ylim=c(0,50))
abline(0, 1,col="red")
plot(true[[2]], mle[[2]], main = "mu", xlab = "True", ylab = "Predicted",xlim=c(0,50),ylim=c(0,50))
abline(0, 1,col="red")
plot(true[[4]] , mle[[4]] , main = "Beta", xlab = "True", ylab ="Predicted",xlim=c(-.6,0.0),ylim=c(-.6,0.0))
abline(0, 1,col="red")
dev.off()


png("data_PLD/plots/cnnltt_mlevsnn.png", width = 480*3*factor, height = 480*factor,res=dpi)
par(mfrow = c(1, 4))
plot(nn.pred_ltt[[1]], mle[[1]], main = "lambda0", xlab = "CNN LTT", ylab = "MLE",xlim=c(0,50),ylim=c(0,50))
abline(0, 1,col="red")
plot(nn.pred_ltt[[2]], mle[[2]], main = "mu", xlab = "CNN LTT", ylab = "MLE",xlim=c(0,50),ylim=c(0,50))
abline(0, 1,col="red")
plot(nn.pred_ltt[[3]] , mle[[3]] , main = "K", xlab = "CNN LTT ", ylab = "MLE",xlim=c(0,400),ylim=c(0,400))
abline(0, 1,col="red")
plot(nn.pred_ltt[[4]] , mle[[4]] , main = "Beta", xlab = "CNN LTT", ylab ="MLE",xlim=c(-.6,0.0),ylim=c(-.6,0.0))
abline(0, 1,col="red")
dev.off()

png("data_DDD/plots/cnn_mlevsnn_wok.png", width = 480*3*factor, height = 480*factor,res=dpi)
par(mfrow = c(1, 4))
plot(nn.pred_cnn[[1]], mle[[1]], main = "lambda0", xlab = "CNN", ylab = "MLE",xlim=c(0,50),ylim=c(0,50))
abline(0, 1,col="red")
plot(nn.pred_cnn[[2]], mle[[2]], main = "mu", xlab = "CNN", ylab = "MLE",xlim=c(0,50),ylim=c(0,50))
abline(0, 1,col="red")
plot(nn.pred_cnn[[3]] , mle[[3]] , main = "K", xlab = "CNN", ylab = "MLE",xlim=c(0,400),ylim=c(0,400))
abline(0, 1,col="red")
plot(nn.pred_cnn[[4]] , mle[[4]] , main = "Beta", xlab = "CNN", ylab ="MLE",xlim=c(-.6,0.0),ylim=c(-.6,0.0))
abline(0, 1,col="red")
dev.off()




# Get the Nnode values for each tree
node_values <- sapply(phylo[test_indices], function(tree) tree$Nnode)

# Group the indices based on the Nnode values
grouped_indices <- split(seq_along(phylo[test_indices]), node_values)



h_cnn <- list()
h_cnnltt<-list()
for( i in 1:length(true_wo_k)){
  h_cnn[[names(true_wo_k[i])]] <- list()
  h_cnnltt[[names(true_wo_k[i])]] <- list()

  
  # Iterate over each group
  for (j in names(grouped_indices)) {
    tree_indices <- grouped_indices[[j]]  # Get the tree indices for the current group
    
    # Get the values for the current name and tree indices
    #values <- true[[i]][tree_indices]
    
    nrmse_cnn <- sqrt(mean((true_wo_k[[i]][tree_indices]-nn.pred_cnn_wo_k[[i]][tree_indices])^2))/abs(mean(true_wo_k[[i]][tree_indices]))
    nrmse_cnnltt <- sqrt(mean((true_wo_k[[i]][tree_indices]-nn.pred_ltt_wo_k[[i]][tree_indices])^2))/abs(mean(true_wo_k[[i]][tree_indices]))

    # Store the values in h
    h_cnn[[i]][[j]] <- nrmse_cnn
    h_cnnltt[[i]][[j]] <- nrmse_cnnltt
  
  }
  
  
}

factor2=2
h_list<-list(h_cnn,h_cnnltt)
h_names<-c("CNN","CNN LTT")
var_names<-c("lambda0","mu","beta N",  "beta P")
j<-1
for (h in h_list){
  
  png(paste("data_PLD/plots/NRMSE_mlw3",h_names[j],".png",sep=""),width=3000,height=1000, res=300)#,width = floor(480*1.3*factor2), height = floor(480*factor2),res=15)
  par(mfrow = c(1, 4))
  for (i in 1:length(h)) {
    nrmse_values <- unlist(h[[i]])  # Get the NRMSE values for the current h[i]
    
    plot(x = as.integer(names(h[[i]])),
         y = nrmse_values,
         xlab = "Tree Size",
         ylab = "NRMSE",
         main = paste(var_names[i]),
          xlim=c(0,800),
          ylim=c(0,5))
    
  }
  dev.off()
  
  
  
  j<-j+1
  
  
}





