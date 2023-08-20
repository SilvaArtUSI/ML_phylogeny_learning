
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
#device<-"cuda"
device<-"cpu"

## Please check names
cblv_crbd <-readRDS( paste("data_clas/phylogeny-reescrbd-10000ld-01-1-e-0-9-cblv.rds",sep=""))
cblv_bisse <-readRDS( paste("data_clas/phylogeny-reesbisse-10000ld-01-1-e-0-9-cblv.rds",sep=""))
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
batch_size <-32

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



n_hidden <- 32
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

best_cnn<- cnn.net(n_input, n_out, n_hidden, n_layer, ker_size) # save best CNN

## Training 


train_batch <- function(b){
  opt$zero_grad()
  output <- cnn(b$x$to(device = device))
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
  output <- cnn(b$x$to(device = device))
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
epoch <- 1
trigger <- 0
patience <- 5
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
best_epoch<-0
best_loss<-10000

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
  
  if (current_loss< best_loss){
    
    torch_save(cnn, paste( "data_clas/models/c03_CNN_32",sep="-"))
    best_epoch<-epoch
    best_loss<-current_loss
    
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
time_cnn <-end_time - start_time
print(time_cnn)

dpi=300


png("data_clas/plots/Loss_curve_CNN.png")
# Plot the loss curve
plot(1:length(train_losses), train_losses, type = "l", col = "blue",
     xlab = "Epoch", ylab = "Loss", main = "Training and Validation Loss",
     ylim = range(c(train_losses, valid_losses)))
lines(1:length(valid_losses), valid_losses, type = "l", col = "red")
legend("topright", legend = c("Training Loss", "Validation Loss"),
       col = c("blue", "red"), lty = 1)
dev.off()
# Close the PNG device

# Plot the accuracy
png("data_clas/plots/Acc_curve_CNN.png")
plot(1:length(train_accuracy), train_accuracy, type = "l", col = "blue",
     xlab = "Epoch", ylab = "Loss", main = "Training and Validation Accuracy",
     ylim = range(c(train_accuracy, valid_accuracy)))
lines(1:length(valid_accuracy), valid_accuracy, type = "l", col = "red")
legend("topright", legend = c("Training Accuracy", "Validation Accuracy"),
       col = c("blue", "red"), lty = 1)

# Close the PNG device
dev.off()



cnn<-torch_load( paste( "data_clas/models/c03_CNN_32",sep="-"))

cnn$to(device=device)


## Evaluation 

#Compute predicted parameters on test set.

cnn$eval()
pred <- vector(mode = "list", length = n_out)
names(pred) <-true_names

Pred_total_list<- list("crbd"= vector() , "bisse" = vector() ,"ddd" = vector() , "pld"= vector())


acc_list <- list("crbd"= 0 , "bisse" = 0 ,"ddd" = 0 , "pld"= 0 ,"total" = 0)
total_list <- list("crbd"= 0 , "bisse" = 0 ,"ddd" = 0 , "pld"= 0, "total"=0)



# Compute accuracy 
coro::loop(for (b in test_dl) {
  output <- cnn(b$x$to(device = device ))
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


result$timemin <- as.numeric(time_cnn)
result$unit <- units(time_cnn)
result$best_epoch<-best_epoch
result$epoch<-epoch


# Print the result
print(result)
#print("Accuray total Testing")
#print(sum(unlist(acc_list))/sum(unlist(total_list)))

write.csv(result, file = "data_clas/results/cnn.csv", row.names = FALSE)


# Plot histograms
png("Plots/hist_cnn2.png")
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




Pred_conf <- lapply(Pred_total_list, function(numbers) {
  tab <- table(factor(numbers, levels = 1:4))
  as.numeric(tab)  # Convert table to numeric vector
})

# Convert Pred_conf to a matrix
confusion_matrix <- do.call(rbind, Pred_conf)

colnames(confusion_matrix) <- c("True CRBD", "True BiSSE", "True DDD", "True PLD")

# Write the data frame to a CSV file
write.csv(confusion_matrix, "data_clas/results/cnn_confmat.csv", row.names = TRUE)



