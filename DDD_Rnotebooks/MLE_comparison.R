
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
n_trees<-100
phylo_nameddd<-"phylogeny-DDD2-nt-100-la0-0-50-mu-0-50-k-20-400-age-1-ddmod-10.rds"
true_nameddd<-"true-param-DDD2-nt-100-la0-0-50-mu-0-50-k-20-400-age-1-ddmod-10.rds"
brts_nameddd<-"brts-DDD2-nt-10000-la0-0-50-mu-0-50-k-20-400-age-1-ddmod-10.rds"

phylo <- readRDS(paste( "data_DDD/",phylo_nameddd,sep =""))
true<-readRDS(paste( "data_DDD/",true_nameddd,sep=""))

true<-true[c(-4)]

mle<-read.csv("data_DDD/MLE_DDD100.csv")
mle$beta<-(mle$mu-mle$lambda0)/mle$K
#true$beta<-(true$mu-true$lambda0)/true$K





max_nodes_roundedcnn <- 1200 #this was encoded with all the phylogenies
max_nodes_roundedcnnltt<-500 #this was encoded separetly


####Realted to cblv encoding

cblv <- generate_encoding_DDD(phylo, tree_size = max_nodes_roundedcnn)
ds_cnn <- convert_encode_to_dataset(cblv, true)
test_ds_cnn  <- ds_cnn(cblv, true)
test_dl_cnn  <- test_ds_cnn  %>% dataloader(batch_size=1, shuffle=FALSE)


#transforming data
df.ltt_new<-new_generate_ltt_dataframe(phylo,max_nodes_roundedcnnltt)
#ds.ltt <- convert_ltt_dataframe_to_dataset(df.ltt_test, out_phyloddd$param, "cnn-ltt")
ds.ltt_new <- new_convert_ltt_dataframe_to_dataset(df.ltt_test,  "cnn-ltt")
#test_ds  <- ds.ltt(df.ltt[, test_indices] , extract_elements(true, test_indices))
ds_eval_ltt  <- ds.ltt_new(df.ltt_new)
data_loader_ltt <- ds_eval_ltt  %>% dataloader(batch_size=1,     shuffle=FALSE)


source("DDD_Rnotebooks/DDD_NN_TEST_MLE.R")


n_out<-3
nn.pred_ltt <- vector(mode = "list", length = n_out)
names(nn.pred_ltt) <- names(c("lambda0", "mu","K" ))

nn.pred_cnn <- vector(mode = "list", length = n_out)
names(nn.pred_cnn ) <- names(c("lambda0","mu","K" ))



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
coro::loop(for (b in data_loader_ltt) {
  #if (model_type == "crbd"){b$x <- b$x$unsqueeze(2)}
  #print(b$x)
  out <- cnn_ltt(b$x$to(device = device))
  pred <- as.numeric(out$to(device = "cpu")) # move the tensor to CPU 
  #true <- as.numeric(b$y)
  for (i in 1:n_out){nn.pred_ltt [[i]] <- c(nn.pred_ltt [[i]], pred[i])}
})


lambda0_ddd <- c(0, 50) # speciation rate
k_ddd	<- c(20,400)

nn.pred_cnn[[1]]<-min_max_unnormalize(nn.pred_cnn[[1]],rang=lambda0_ddd)
nn.pred_cnn[[2]]<-min_max_unnormalize(nn.pred_cnn[[2]],rang=lambda0_ddd)
nn.pred_cnn[[3]]<-min_max_unnormalize(nn.pred_cnn[[3]],rang=k_ddd)
nn.pred_cnn[[4]]<-(nn.pred_cnn[[2]] - nn.pred_cnn[[1]])/ nn.pred_cnn[[3]]


nn.pred_ltt[[1]]<-min_max_unnormalize(nn.pred_ltt[[1]],rang=lambda0_ddd)
nn.pred_ltt[[2]]<-min_max_unnormalize(nn.pred_ltt[[2]],rang=lambda0_ddd)
nn.pred_ltt[[3]]<-min_max_unnormalize(nn.pred_ltt[[3]],rang=k_ddd)
nn.pred_ltt[[4]]<-(nn.pred_ltt[[2]] - nn.pred_ltt[[1]])/ nn.pred_ltt[[3]]

true$beta<-(true$mu-true$lambda0)/true$K



dpi=300
factor=floor(dpi/150)


#Plots
###CNN
png("data_DDD/plots/cnn_estimated_100.png", width = 480*3*factor, height = 480*factor,res=dpi)
par(mfrow = c(1, 4))
plot(true[[1]], nn.pred_cnn[[1]], main = "lambda0", xlab = "True", ylab = "Predicted")
abline(0, 1,col="red")
plot(true[[2]], nn.pred_cnn[[2]], main = "mu", xlab = "True", ylab = "Predicted")
abline(0, 1,col="red")
plot(true[[3]] , nn.pred_cnn[[3]] , main = "K", xlab = "True", ylab = "Predicted")
abline(0, 1,col="red")
plot(true[[4]]  ,  nn.pred_cnn[[4]] , main = "Beta", xlab = "True", ylab ="Predicted",xlim=c(-.8,0),ylim=c(-.8,0))
abline(0, 1,col="red")
dev.off()

#Plots
###CNNLTT
png("data_DDD/plots/cnnltt_estimated_100.png", width = 480*3*factor, height = 480*factor,res=dpi)
par(mfrow = c(1, 4))
plot(true[[1]], nn.pred_ltt[[1]], main = "lambda0", xlab = "True", ylab = "Predicted")
abline(0, 1,col="red")
plot(true[[2]], nn.pred_ltt[[2]], main = "mu", xlab = "True", ylab = "Predicted")
abline(0, 1,col="red")
plot(true[[3]] , nn.pred_ltt[[3]] , main = "K", xlab = "True", ylab = "Predicted")
abline(0, 1,col="red")
plot(true[[4]]  ,  nn.pred_ltt[[4]] , main = "Beta", xlab = "True", ylab ="Predicted",xlim=c(-.8,0),ylim=c(-.8,0))
abline(0, 1,col="red")
dev.off()

#Plots
###MLE
png("data_DDD/plots/mle_estimated_100.png", width = 480*3*factor, height = 480*factor,res=dpi)
par(mfrow = c(1, 4))
plot(true[[1]], mle[[1]], main = "lambda0", xlab = "NN", ylab = "MLE",xlim=c(0,50),ylim=c(0,50))
abline(0, 1,col="red")
plot(true[[2]], mle[[2]], main = "mu", xlab = "NN", ylab = "MLE",xlim=c(0,50),ylim=c(0,50))
abline(0, 1,col="red")
plot(true[[3]] , mle[[3]] , main = "K", xlab = "NN", ylab = "MLE",xlim=c(0,400),ylim=c(0,400))
abline(0, 1,col="red")
plot(true[[4]] , mle[[4]] , main = "Beta", xlab = "NN", ylab ="MLE",xlim=c(-.6,0.0),ylim=c(-.6,0.0))
abline(0, 1,col="red")
dev.off()


png("data_DDD/plots/cnnltt_mlevsnn.png", width = 480*3*factor, height = 480*factor,res=dpi)
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

png("data_DDD/plots/cnn_mlevsnn.png", width = 480*3*factor, height = 480*factor,res=dpi)
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




column_names <- c("lambda0", "mu", "K", "Beta_N")

error_mle <- vector(mode = "list", length = 4)
error_cnn <- vector(mode = "list", length = 4)
error_cnnltt <- vector(mode = "list", length = 4)


for (i in 1:4){
  error_mle[[i]]<-true[[i]]-mle[[i]]
  error_cnnltt[[i]] <- true[[i]]-nn.pred_ltt[[i]]
  error_cnn[[i]] <- true[[i]]-nn.pred_cnn[[i]]
  
  }

# Convert list to dataframe with custom column names
df.error_mle <- data.frame(error_mle,check.names = FALSE)
colnames(df.error_mle) = column_names

df.error_cnnltt <- data.frame(error_cnnltt,check.names = FALSE)
colnames(df.error_cnnltt) = column_names

df.error_cnn<- data.frame(error_cnn,check.names = FALSE)
colnames(df.error_cnn) = column_names


# Print the resulting dataframe
print(my_dataframe)

df.error_nn$method = "NN"
df.error_mle$method = "MLE"

lg = which(df.error_mle$lambda0 <  -100000)
infinitesk = which(mle$K > 100000)
rmoutliers= c(-lg,-infinitesk)

df.error=  rbind(df.error_nn[rmoutliers,],df.error_mle[rmoutliers,])


ggplot(df.error) + geom_point(aes(x=lambda0,y=mu,colour=method))+theme_minimal()






# Melt the data frame for easier plotting
melted_data_cnnltt <- reshape2::melt(df.error_cnnltt[rmoutliers,])
melted_data_cnn <- reshape2::melt(df.error_cnn[rmoutliers,])
melted_data_mle <- reshape2::melt(df.error_mle[rmoutliers,])

# Create a custom color palette
my_palette <- c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3")

# Create a violin plot with facets for each column
cnnltt_vio<-ggplot(melted_data_cnnltt, aes(x = variable, y = value, fill = variable)) +
  geom_violin() +
  geom_boxplot(width = 0.2, color = "black", fill = "white") +
  labs(x = "", y = "Error") +
  facet_wrap(~ variable, scales = "free",nrow = 1, ncol = 4) +
  scale_fill_manual(values = my_palette) +  # Apply custom color palette
  theme_minimal() +
  guides(fill = "none") 


cnn_vio<-ggplot(melted_data_cnn, aes(x = variable, y = value, fill = variable)) +
  geom_violin() +
  geom_boxplot(width = 0.2, color = "black", fill = "white") +
  labs(x = "", y = "Error") +
  facet_wrap(~ variable, scales = "free",nrow = 1, ncol = 4) +
  scale_fill_manual(values = my_palette) +  # Apply custom color palette
  theme_minimal() +
  guides(fill = "none") 



mle_vio<-ggplot(melted_data_mle, aes(x = variable, y = value, fill = variable)) +
  geom_violin() +
  geom_boxplot(width = 0.2, color = "black", fill = "white") +
  labs(x = "", y = "Error") +
  facet_wrap(~ variable, scales = "free",nrow = 1, ncol = 4) +
  scale_fill_manual(values = my_palette) +  # Apply custom color palette
  theme_minimal() +
  guides(fill = "none") 

# Save the plot to a PNG file

png("data_DDD/plots/violin_plotcnnltt.png", width = 1800, height = 1000,res=300)
print(cnnltt_vio)
dev.off()

png("data_DDD/plots/violin_plotcnn.png", width = 1800, height = 1000,res=300)
print(cnn_vio)
dev.off()

png("data_DDD/plots/violin_plotmle.png", width = 1800, height = 1000,res=300)
print(mle_vio)
dev.off()




  