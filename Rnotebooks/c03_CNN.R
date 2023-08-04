
#Convolutinal Neural Network


library(torch)
library(svMisc)
library(torch)
library(dplyr)
library(caret)
#print(getwd())
source("R/infer-general-functions.R")#Contains functions for generating the phylogenetic trees ,plotting, computing rates.
source("R/neural-network-functions.R")
source("R/convert-phylo-to-sumstat.R")#contains the functions to compute the summary statistics of a phylogenetic tree(tips,depth)
source("R/convert-phylo-to-cblv.R") #contains the function to encode a phylogenetic tree with the "Compact Bijective Ladderized
source("R/new_funcs.R")


max_nodes_rounded<-1200 #computed in previous codes
n_trees<-10000
n_mods<-4
device<-"cpu"



cblv_crbd <-readRDS( paste("data_clas/phylogeny-crbd10000ld-01-1-e-0-9-cblv.rds",sep=""))
cblv_bisse <-readRDS( paste("data_clas/phylogeny-bisse-10000ld-.01-1.0-q-.01-.1-cblv.rds",sep=""))
cblv_ddd <-readRDS( paste("data_clas/phylogeny-DDD2-nt-10000-la0-0-50-mu-0-50-k-20-400-age-1-ddmod-10-cblv.rds",sep=""))
cblv_pld <-readRDS( paste("data_clas/phylogeny-pld-nt-10000-la0-0-50-mu-0-50-k-20-400-age-1-ddmod-10-cblv.rds",sep=""))

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
cblv_total<-cbind(cblv_crbd,cblv_bisse,cblv_ddd,cblv_pld)


true_total <- lapply(1:4, function(i) c(true_crbd[[i]], true_bisse[[i]], true_ddd[[i]], true_pld[[i]]))
names(true_total)<-true_names

set.seed(113)
# Define size of datasets.
total_data_points <- n_trees*n_mods
subset_size <-  n_trees*n_mods  # Specify the size of the subset

n_train    <- floor(subset_size * .9)
n_valid    <- floor(subset_size * .05)
n_test     <- subset_size - n_train - n_valid
#batch_size <- batch_size <- min(subset_size*.01, 10)
batch_size <-64

# Pick the phylogenies randomly.

ds <- convert_encode_to_dataset(cblv_total, true_total)

# Pick the random subset of data points.
subset_indices <- sample(1:ncol(cblv_total), subset_size)

# Split the subset into train, validation, and test indices.
train_indices <- subset_indices[1:n_train]
valid_indices <- subset_indices[(n_train + 1):(n_train + n_valid)]
test_indices  <- subset_indices[(n_train + n_valid + 1):subset_size]


#train_indices <- sample(1:ncol(cblv), n_train)
#not_train_indices <- setdiff(1:ncol(cblv), train_indices)
#valid_indices <- sample(not_train_indices, n_valid)
#test_indices  <- setdiff(not_train_indices, valid_indices)

# Create the datasets.
train_ds <- ds(cblv_total[1:nrow(cblv_total), train_indices], 
               extract_elements(true_total, train_indices))
valid_ds <- ds(cblv_total[1:nrow(cblv_total), valid_indices], 
               extract_elements(true_total, valid_indices))
test_ds  <- ds(cblv_total[1:nrow(cblv_total), test_indices], 
               extract_elements(true_total, test_indices))

# Create the dataloader.
train_dl <- train_ds %>% dataloader(batch_size=batch_size, shuffle=TRUE)
valid_dl <- valid_ds %>% dataloader(batch_size=batch_size, shuffle=FALSE)
test_dl  <- test_ds  %>% dataloader(batch_size=1, shuffle=FALSE)



n_hidden <- 8
n_layer  <- 4
ker_size <- 10
n_input  <- nrow(cblv_total)
n_out    <- length(true_total)
p_dropout <- 0.01


# Build the CNN

cnn.net <- nn_module(
  
  "corr-cnn",
  
  initialize = function(n_input, n_out, n_hidden, n_layer, ker_size) {
    self$conv1 <- nn_conv1d(in_channels = 1, out_channels = n_hidden, kernel_size = ker_size)
    self$conv2 <- nn_conv1d(in_channels = n_hidden, out_channels = 2*n_hidden, kernel_size = ker_size)
    self$conv3 <- nn_conv1d(in_channels = 2*n_hidden, out_channels = 4*n_hidden, kernel_size = ker_size)
    self$conv4 <- nn_conv1d(in_channels = 4*n_hidden, out_channels = 8*n_hidden, kernel_size = ker_size)
    n_flatten  <- compute_dim_ouput_flatten_cnn(n_input, n_layer, ker_size)
    self$fc1   <- nn_linear(in_features = n_flatten * (8*n_hidden), out_features = 100)
    self$fc2   <- nn_linear(in_features = 100, out_features = n_out)
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
      
      self$conv4() %>%
      nnf_relu() %>%
      nnf_dropout(p = p_dropout) %>%
      nnf_avg_pool1d(2) %>%
      
      torch_flatten(start_dim = 2) %>%
      self$fc1() %>%
      nnf_dropout(p = p_dropout) %>%
      nnf_relu() %>%
      
      self$fc2()
  }
)

cnn <- cnn.net(n_input, n_out, n_hidden, n_layer, ker_size) # create CNN
cnn$to(device = device) # Move it to the choosen GPU
opt <- optim_adam(params = cnn$parameters) # optimizer 

## Training 


train_batch <- function(b){
  opt$zero_grad()
  output <- cnn(b$x$to(device = device))
  target <- b$y$to(device = device)
  loss <- nnf_cross_entropy(output, target)
  loss$backward()
  opt$step()
  loss$item()
  
  
  # Compute accuracy
  max_indices1 <- apply(output, 1, which.max)
  max_indices2 <- apply(target, 1, which.max)
  acc <- sum(max_indices1 == max_indices2)
  total <- length(max_indices1)
  
  return(list(loss = loss$item(), accuracy = acc, total = total))
}

valid_batch <- function(b) {
  output <- cnn(b$x$to(device = device))
  target <- b$y$to(device = device)
  loss <- nnf_cross_entropy(output, target)
  loss$item()
  
  
  # Compute accuracy
  max_indices1 <- apply(output, 1, which.max)
  max_indices2 <- apply(target, 1, which.max)
  acc <- sum(max_indices1 == max_indices2)
  total <- length(max_indices1)
  
  return(list(loss = loss$item(), accuracy = acc, total = total))
}

# Initialize parameters for the training loop 
epoch <- 1
trigger <- 0
patience <- 10
n_epochs <- 100
last_loss <- 100


# Training loop 


train_losses <- list()
valid_losses <- list()

train_accuracy <- list()
valid_accuracy <- list()

train_plots <- list()
valid_plots <- list()
start_time <-  Sys.time()
while (epoch < n_epochs & trigger < patience) {
  
  # Training part 
  cnn$train()
  
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
  
  # Evaluation part 
  cnn$eval()
  valid_loss <- c()
  valid_accu <- c()
  coro::loop(for (b in test_dl) { # loop over batches 
    loss <- valid_batch(b)
    valid_loss <- c(valid_loss, loss$loss)
    valid_accu <- c(valid_accu, loss$accuracy/loss$total)
  })
  current_loss <- mean(valid_loss)
  current_accu <- mean(valid_accu)
  
  if (current_loss > last_loss){trigger <- trigger + 1}
  else{
    trigger   <- 0
    last_loss <- current_loss
  }
  
  # Print Epoch and value of Loss function
  cat(sprintf("epoch %0.3d/%0.3d - valid - loss: %3.5f - accuracy: %3.5f  \n",
              epoch, n_epochs, current_loss,current_accu ))
  
  train_losses <- c(train_losses, mean_tl)
  valid_losses <- c(valid_losses, current_loss)
  
  train_accuracy <-c(train_accuracy,mean_ta)
  valid_accuracy <-c(valid_accuracy,current_accu)
  
  epoch <- epoch + 1 
}
end_time <- Sys.time()
time <-end_time - start_time



png("loss_curve_cnn.png")
# Plot the loss curve
plot(1:length(train_losses), train_losses, type = "l", col = "blue",
     xlab = "Epoch", ylab = "Loss", main = "Training and Validation Loss",
     ylim = range(c(train_losses, valid_losses)))
lines(1:length(valid_losses), valid_losses, type = "l", col = "red")
legend("topright", legend = c("Training Loss", "Validation Loss"),
       col = c("blue", "red"), lty = 1)

# Close the PNG device
dev.off()


png("acc_curve_cnn.png")
# Plot the accuracy
plot(1:length(train_accuracy), train_accuracy, type = "l", col = "blue",
     xlab = "Epoch", ylab = "Loss", main = "Training and Validation Accuracy",
     ylim = range(c(train_accuracy, valid_accuracy)))
lines(1:length(valid_accuracy), valid_accuracy, type = "l", col = "red")
legend("topright", legend = c("Training Accuracy", "Validation Accuracy"),
       col = c("blue", "red"), lty = 1)

# Close the PNG device
dev.off()




torch_save(cnn, paste( "models/c01_CNN_1st_try",sep="-"))
cat(paste("\n Model cnn saved", sep = ""))
cat("\nSaving model... Done.")

cnn<-torch_load( paste( "models/c01_CNN_1st_try",sep="-"))

cnn$to(device=device)


## Evaluation 

#Compute predicted parameters on test set.

cnn$eval()
pred <- vector(mode = "list", length = n_out)
names(pred) <-true_names

acc_list <- list("crbd"= 0 , "bisse" = 0 ,"ddd" = 0 , "pld"= 0 ,"total" = 0)
total_list <- list("crbd"= 0 , "bisse" = 0 ,"ddd" = 0 , "pld"= 0, "total"=0)

# Compute accuracy 
coro::loop(for (b in test_dl) {
  output <- cnn(b$x$to(device = device))
  target <-  b$y$to(device = device)
  
  max_indices1 <- apply(output, 1, which.max)
  max_indices2 <- apply(target, 1, which.max)
  acc <- sum(max_indices1 == max_indices2)
  total <- length(max_indices1)
  
  if(max_indices2==1){
    acc_list$crbd=acc_list$crbd+acc
    total_list$crbd=total_list$crbd+total}
  
  if(max_indices2==2){
    acc_list$bisse=acc_list$bisse+acc
    total_list$bisse=total_list$bisse+total}
  
  if(max_indices2==3){
    acc_list$ddd=acc_list$ddd+acc
    total_list$ddd=total_list$ddd+total}
  
  if(max_indices2==4){
    acc_list$pld=acc_list$pld+acc
    total_list$pld=total_list$pld+total}
  
  
  acc_list$total= acc_list$total+acc
  total_list$total=total_list$total+total
  
  
  
})


result <- Map("/", acc_list, total_list)

# Print the result
print(result)
#print("Accuray total Testing")
#print(sum(unlist(acc_list))/sum(unlist(total_list)))

write.csv(result, file = "Testing_results/cnn.csv", row.names = FALSE)

Now that you have the predicted parameters you can, for instance, 
plot the predicted value by the neural network vs. the true values.

## Plots Prediction vs Predicted
```{R, Plots Prediciton vs Predicted}
# Prepare plot 
par(mfrow = c(1, 3))

plot(true[[1]][test_indices], pred[[1]], main = "lambda0", xlab = "True", ylab = "Predicted")
abline(0, 1,col="red")

plot(true[[2]][test_indices], pred[[2]], main = "mu", xlab = "True", ylab = "Predicted")
abline(0, 1,col="red")

plot(true[[3]][test_indices] * 100, pred[[3]] * 100, main = "K", xlab = "True", ylab = "Predicted")
abline(0, 1,col="red")
```

```{R,computing RMSE}

mse_l <- list()

for (i in 1:length(true))
{
  mse<- sqrt(mean((true[[i]][test_indices]-pred[[i]])^2))/mean(true[[i]][test_indices])
  mse_l<-c(mse_l, mse)
}

names(mse_l) <- names(true)

mse_l


```


```{R,NRME against bucket, Only on testing}

phylo_name<-fname_ddd(n_trees,lambda0,mu,k,crown_age,dd_model)
phylo <- readRDS( paste("data/phylogeny-",model,"-",phylo_name,".rds",sep=""))

# Get the Nnode values for each tree
node_values <- sapply(phylo[test_indices], function(tree) tree$Nnode)

# Group the indices based on the Nnode values
grouped_indices <- split(seq_along(phylo[test_indices]), node_values)


h <- list()
for( i in 1:length(true)){
  h[[names(true[i])]] <- list()
  
  # Iterate over each group
  for (j in names(grouped_indices)) {
    tree_indices <- grouped_indices[[j]]  # Get the tree indices for the current group
    
    # Get the values for the current name and tree indices
    #values <- true[[i]][tree_indices]
    
    nrmse <- sqrt(mean((true[[i]][test_indices][tree_indices]-pred[[i]][tree_indices])^2))/mean(true[[i]][test_indices][tree_indices])
    
    # Store the values in h
    h[[i]][[j]] <- nrmse
    
  }
  
  
}

```


## Plotting Tree Size vs NRMSE 
```{R, Plotting NRMSE Tree Size }
# Scatter plot for each h[i]
for (i in 1:length(h)) {
  nrmse_values <- unlist(h[[i]])  # Get the NRMSE values for the current h[i]
  
  # Create a scatter plot
  plot(x = as.integer(names(h[[i]])),
       y = nrmse_values,
       xlab = "Number of Nodes in Tree",
       ylab = "NRMSE",
       main = paste("Scatter Plot of NRMSE -", names(h)[i]))
}


```
