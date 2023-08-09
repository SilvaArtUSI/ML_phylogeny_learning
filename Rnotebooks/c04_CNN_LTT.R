# Importing Libraries and Sources

library(torch)
library(luz)
library(torch)
library(svMisc)
library(torch)
library(dplyr)
library(caret)
source("R/neural-network-functions.R")
source("R/infer-general-functions.R")
source("R/convert-phylo-to-cblv.R")
source("R/phylo-inference-ml.R")
source("R/new_funcs.R")





n_trees <- 10000# number of trees to generate
device <- "cuda"
nn_type <- "cnn-ltt"
max_nodes_rounded<-1200
n_mods<-4
generateltt<-FALSE



###Reading LTT



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


if (generateltt==TRUE){
  
phylo_crbd <- readRDS( paste("data_clas/phylogeny-crbd10000ld-01-1-e-0-9.rds", sep=""))
phylo_bisse <- readRDS( paste("data_clas/phylogeny-bisse-10000ld-.01-1.0-q-.01-.1.rds", sep=""))
phylo_ddd <- readRDS( paste("data_clas/phylogeny-DDD2-nt-10000-la0-0-50-mu-0-50-k-20-400-age-1-ddmod-10.rds", sep=""))
phylo_pld <- readRDS( paste("data_clas/phylogeny-pld-nt-10000-la0-0-50-mu-0-50-k-20-400-age-1-ddmod-10.rds", sep=""))
 
phylo<-c(phylo_crbd,phylo_bisse,phylo_ddd,phylo_pld)


start_time <- Sys.time()

df.ltt <- generate_ltt_dataframe(phylo, max_nodes_rounded, true)$
  
  
saveRDS(df.ltt, paste("data_clas/phylogeny-all-dfltt.rds", sep=""))  

end_time <- Sys.time()
print(end_time - start_time)

rm(phylo)
rm(phylo_crbd)
rm(phylo_bisse)
rm(phylo_ddd)
rm(phylo_pld)

}else{
  
  df.ltt<-readRDS(paste("data_clas/phylogeny-all-dfltt.rds", sep=""))
  
}



# Parameters of the NN's training

set.seed(113)
total_data_points<-n_trees*n_mods
subset_size <- 10000  # Specify the size of the subset

n_train    <- floor(subset_size * .9)
n_valid    <- floor(subset_size * .05)
n_test     <- subset_size - n_train - n_valid
batch_size <- 64

# Pick the phylogenies randomly.
ds.ltt <- convert_ltt_dataframe_to_dataset(df.ltt, true, nn_type)


# Pick the random subset of data points.
subset_indices <- sample(1:total_data_points, subset_size)

# Split the subset into train, validation, and test indices.
train_indices <- subset_indices[1:n_train]
valid_indices <- subset_indices[(n_train + 1):(n_train + n_valid)]
test_indices  <- subset_indices[(n_train + n_valid + 1):subset_size]


#n_train    <- .9 *n_trees
#n_valid    <- .05*n_trees
#n_test     <- n_trees - n_train - n_valid
#n_epochs   <- 100
#batch_size <- 64
#patience   <- 10

# Creation of the train, valid and test dataset
#train_indices     <- sample(1:n_trees, n_train)
#not_train_indices <- setdiff(1:n_trees, train_indices)
#valid_indices     <- sample(not_train_indices, n_valid)
#test_indices      <- setdiff(not_train_indices, valid_indices)

train_ds <- ds.ltt(df.ltt[, train_indices], extract_elements(true, train_indices))
valid_ds <- ds.ltt(df.ltt[, valid_indices], extract_elements(true, valid_indices))
test_ds  <- ds.ltt(df.ltt[, test_indices] , extract_elements(true, test_indices))


# Creation of the dataloader 
train_dl <- train_ds %>% dataloader(batch_size=batch_size, shuffle=TRUE)
valid_dl <- valid_ds %>% dataloader(batch_size=batch_size, shuffle=FALSE)
test_dl  <- test_ds  %>% dataloader(batch_size=1,          shuffle=FALSE)


n_hidden  <- 16
n_layer   <- 3
ker_size  <- 5
p_dropout <- 0.01
n_input   <- max_nodes_rounded
n_out     <- n_mods

# Build the CNN

cnn.net <- nn_module(
  
  "corr-cnn",
  
  initialize = function(n_input, n_out, n_hidden, n_layer, ker_size, p_dropout) {
    self$conv1 <- nn_conv1d(in_channels = 1, out_channels = n_hidden, kernel_size = ker_size)
    self$conv2 <- nn_conv1d(in_channels = n_hidden, out_channels = 2*n_hidden, kernel_size = ker_size)
    self$conv3 <- nn_conv1d(in_channels = 2*n_hidden, out_channels = 2*2*n_hidden, kernel_size = ker_size)
    n_flatten <- compute_dim_ouput_flatten_cnn(n_input, n_layer, ker_size)
    self$fc1 <- nn_linear(in_features = n_flatten * (2*2*n_hidden), out_features = 100)
    self$fc2 <- nn_linear(in_features = 100, out_features = n_out)
  },
  
  forward = function(x) {
    x %>% 
      self$conv1() %>%
      nnf_relu() %>%
      nnf_dropout(p = p_dropout) %>%
      nnf_avg_pool1d(2) %>%
      
      self$conv2() %>%
      nnf_relu() %>%
      nnf_dropout(p = p_dropout) %>%
      nnf_avg_pool1d(2) %>%
      
      self$conv3() %>%
      nnf_relu() %>%
      nnf_dropout(p = p_dropout) %>%
      nnf_avg_pool1d(2) %>%
      
      torch_flatten(start_dim = 2) %>%
      self$fc1() %>%
      nnf_relu() %>%
      nnf_dropout(p = p_dropout) %>%
      
      self$fc2()
  }
)



cnn_ltt <- cnn.net(n_input, n_out, n_hidden, n_layer, ker_size, p_dropout) # create CNN
cnn_ltt$to(device = device) # Move it to the choosen GPU

# Prepare training 

opt <- optim_adam(params = cnn_ltt$parameters) # optimizer 



train_batch <- function(b){
  opt$zero_grad()
  #if (model_type == "crbd"){b$x <- b$x$unsqueeze(2)}
  output <- cnn_ltt(b$x$to(device = device))
  target <- b$y$to(device = device)
  loss <- nnf_cross_entropy(output, target)
  loss$backward()
  opt$step()
  loss$item()
  
  max_indices1 <- torch_argmax(output, dim = 2,keepdim =FALSE)  # Find the predicted class labels
  max_indices2 <- torch_argmax(target, dim = 2, keepdim =FALSE)  # Find the true class labels
  
  #print(max_indices1)
  
  acc <- torch_sum(max_indices1 == max_indices2)
  total <- length(max_indices1)
  
  return(list(loss = loss$item(), accuracy = acc$item(), total = total))
  
  

}

valid_batch <- function(b) {
  #if (model_type == "crbd"){b$x <- b$x$unsqueeze(2)}
  output <- cnn_ltt(b$x$to(device = device))
  target <- b$y$to(device = device)
  loss <- nnf_cross_entropy(output, target)
  loss$item()
  
  
  max_indices1 <- torch_argmax(output, dim = 2,keepdim =FALSE)  # Find the predicted class labels
  max_indices2 <- torch_argmax(target, dim = 2, keepdim =FALSE)  # Find the true class labels
  
  #print(max_indices1)
  
  acc <- torch_sum(max_indices1 == max_indices2)
  total <- length(max_indices1)
  
  return(list(loss = loss$item(), accuracy = acc$item(), total = total))
}



# Initialize parameters for the training loop 
epoch     <- 1
trigger   <- 0 
last_loss <- 100
n_epochs<-100
patience<-10
best_loss<-10000
best_epoch<-0



# Training loop 

train_losses <- list()
valid_losses <- list()

train_accuracy <- list()
valid_accuracy <- list()

train_plots <- list()
valid_plots <- list()
start_time <-  Sys.time()

while (epoch < n_epochs & trigger < patience) {
  
  # Training 
  cnn_ltt$train()
  train_loss <- c()
  train_accu <- c()
  coro::loop(for (b in train_dl) { # loop over batches 
    loss <- train_batch(b)
    train_loss <- c(train_loss, loss$loss)
    train_accu <- c(train_accu, loss$accuracy/loss$total)
  })
  mean_tl<-mean(train_loss)
  mean_ta<-mean(train_accu)
  
  # Print Epoch and value of Loss function 
  cat(sprintf("epoch %0.3d/%0.3d - train - loss: %3.5f - accuracy: %3.5f \n",
              epoch, n_epochs, mean_tl, mean_ta))
  
  # Validation 
  cnn_ltt$eval()
  valid_loss <- c()
  valid_accu <- c()
  coro::loop(for (b in test_dl) { # loop over batches 
    loss <- valid_batch(b)
    valid_loss <- c(valid_loss, loss$loss)
    valid_accu <- c(valid_accu, loss$accuracy/loss$total)
  })
  current_loss <- mean(valid_loss)
  current_accu <- mean(valid_accu)
  
  # Early Stopping 
  if (current_loss > last_loss){trigger <- trigger + 1} 
  else{
    trigger   <- 0
    last_loss <- current_loss
  }
  if (current_loss< best_loss){
    
    torch_save(cnn_ltt, paste( "models/CNNLTT_FIRST_TRy2",sep="-"))
    best_epoch<-epoch
    best_loss<-current_loss
    
  }
  
  # Print Epoch and value of Loss function
  cat(sprintf("epoch %0.3d/%0.3d - valid - loss: %3.5f - accuracy: %3.5f  \n",
              epoch, n_epochs, current_loss,current_accu ))
  
  train_losses <- c(train_losses, mean(train_loss))
  valid_losses <- c(valid_losses, current_loss)
  
  train_accuracy <-c(train_accuracy,mean_ta)
  valid_accuracy <-c(valid_accuracy,current_accu)
  
  epoch <- epoch + 1 
  
  
}
end_time <- Sys.time()

print(end_time - start_time)

time_cnnltt<-end_time - start_time


png("Plots/loss_curve_cnnltt.png")
# Plot the loss curve
plot(1:length(train_losses), train_losses, type = "l", col = "blue",
     xlab = "Epoch", ylab = "Loss", main = "Training and Validation Loss",
     ylim = range(c(train_losses, valid_losses)))
lines(1:length(valid_losses), valid_losses, type = "l", col = "red")
legend("topright", legend = c("Training Loss", "Validation Loss"),
       col = c("blue", "red"), lty = 1)

# Close the PNG device
dev.off()


png("Plots/acc_curve_cnnltt.png")
# Plot the accuracy
plot(1:length(train_accuracy), train_accuracy, type = "l", col = "blue",
     xlab = "Epoch", ylab = "Loss", main = "Training and Validation Accuracy",
     ylim = range(c(train_accuracy, valid_accuracy)))
lines(1:length(valid_accuracy), valid_accuracy, type = "l", col = "red")
legend("topright", legend = c("Training Accuracy", "Validation Accuracy"),
       col = c("blue", "red"), lty = 1)

# Close the PNG device
dev.off()






cnn_ltt<-torch_load( paste( "models/CNNLTT_FIRST_TRy",patience,sep="-"))
cnn_ltt$to(device=device)

cnn_ltt$eval()
pred <- vector(mode = "list", length = n_out)
names(pred) <-true_names

Pred_total_list<- list("crbd"= vector() , "bisse" = vector() ,"ddd" = vector() , "pld"= vector())


acc_list <- list("crbd"= 0 , "bisse" = 0 ,"ddd" = 0 , "pld"= 0 ,"total" = 0)
total_list <- list("crbd"= 0 , "bisse" = 0 ,"ddd" = 0 , "pld"= 0, "total"=0)



# Compute accuracy 
coro::loop(for (b in test_dl) {
  output <- cnn_ltt(b$x$to(device = device ))
  target <-  b$y$to(device = device)
  
  output<-torch_tensor(output,device = 'cpu')
  target<-torch_tensor(target,device = 'cpu')
  
  
  max_indices1 <- apply(output, 1, which.max)
  max_indices2 <- apply(target, 1, which.max)
  acc <- sum(max_indices1 == max_indices2)
  total <- length(max_indices1)
  
  if(max_indices2==1){
    acc_list$crbd=acc_list$crbd+acc
    total_list$crbd=total_list$crbd+total
    Pred_total_list$crbd <-c(Pred_total_list$crbd,max_indices1)
    
  }
  
  if(max_indices2==2){
    acc_list$bisse=acc_list$bisse+acc
    total_list$bisse=total_list$bisse+total
    Pred_total_list$bisse <-c(Pred_total_list$bisse,max_indices1)
    
  }
  
  if(max_indices2==3){
    acc_list$ddd=acc_list$ddd+acc
    total_list$ddd=total_list$ddd+total
    Pred_total_list$ddd <-c(Pred_total_list$ddd,max_indices1)
  }
  
  if(max_indices2==4){
    acc_list$pld=acc_list$pld+acc
    total_list$pld=total_list$pld+total
    Pred_total_list$pld <-c(Pred_total_list$pld,max_indices1)
  }
  
  
  acc_list$total= acc_list$total+acc
  total_list$total=total_list$total+total
  
  
  
})



result <- Map("/", acc_list, total_list)


result$timemin <- as.numeric(time_cnnltt)
result$best_epoch<-best_epoch
result$epoch<-epoch


# Print the result
print(result)

## Plots Prediction vs Predicted



write.csv(result, file = "Testing_results/cnnltt.csv", row.names = FALSE)


# Plot histograms
png("Plots/hist_cnnltt2.png")
par(mfrow = c(2, 2)) # Adjust the layout based on your preferences

categories <- c("crbd", "bisse", "ddd", "pld")

for (category in categories) {
  hist(Pred_total_list[[category]],
       main = paste("Histogram for", category),
       xlab = "Prediction",
       xlim = c(-0.5, 4.5),  # Adjust xlim to center bars
       breaks = -0.5:4.5)   # Adjust breaks to center bars
}

dev.off()




