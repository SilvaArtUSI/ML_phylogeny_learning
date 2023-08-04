######
#####
#####


#Libraries

library(svMisc)
library(torch)
library(dplyr)
library(caret)

source("R/infer-general-functions.R")#Contains functions for generating the phylogenetic trees ,plotting, computing rates.
source("R/convert-phylo-to-sumstat.R")#contains the functions to compute the summary statistics of a phylogenetic tree(tips,depth)
source("R/convert-phylo-to-cblv.R") #contains the function to encode a phylogenetic tree with the "Compact Bijective Ladderized
source("R/new_funcs.R")



##Loading data


n_trees<-10000
num_mods<-4
device="cpu"

###Loading Summary Statistics
##Rename as necessary

SS_check_NA_stat<-c(-7,	-8,	-9,	-10,	-11,	-12,	-13,	-14,	-15,	-16,	-17,	-18,	-19,	
   -20,	-21,	-22,	-23,	-24,	-35,	-36,	-37,	-38,	-39,	-40,	-41,	
   -42,	-43,	-44)


###CBRD###
sumstat_crbd <- readRDS( paste("data_clas/phylogeny-crbd-",n_trees,"ld-01-1-e-0-9.rds-sumstat.rds", sep=""))
sumstat_crbd <- sumstat_crbd[c(-86,-85)]
sumstat_crbd <- sumstat_crbd [SS_check_NA_stat]

sumstat_crbd$crbd <-2
sumstat_crbd$bisse <-1
sumstat_crbd$ddd <-1
sumstat_crbd$pld <-1 


###Bisse###
sumstat_bisse <-readRDS( paste("data_clas/phylogeny-bisse-",n_trees,"ld-.01-1.0-q-.01-.1.rds-sumstat.rds", sep=""))
sumstat_bisse <- sumstat_bisse[c(-90,-89,-88,-87,-86,-85)] 
sumstat_bisse <- sumstat_bisse [SS_check_NA_stat]

sumstat_bisse$crbd <-1
sumstat_bisse$bisse <-2
sumstat_bisse$ddd <-1
sumstat_bisse$pld <-1 



###DDD###
sumstat_ddd <-readRDS( paste("data_clas/phylogeny-DDD2-nt-10000-la0-0-50-mu-0-50-k-20-400-age-1-ddmod-10-sumstat.rds", sep=""))
sumstat_ddd <- sumstat_ddd[c(-88,-87,-86,-85)] 
sumstat_ddd <- sumstat_ddd[SS_check_NA_stat]

sumstat_ddd$crbd <-1
sumstat_ddd$bisse <-1
sumstat_ddd$ddd <-2
sumstat_ddd$pld <-1 


###PLD###
sumstat_pld <-readRDS( paste("data_clas/phylogeny-pld-nt-10000-la0-0-50-mu-0-50-k-20-400-age-1-ddmod-10-sumstat.rds", sep=""))
sumstat_pld <- sumstat_pld[c(-88,-87,-86,-85)] 
sumstat_pld <- sumstat_pld[SS_check_NA_stat]

sumstat_pld$crbd <-1
sumstat_pld$bisse <-1
sumstat_pld$ddd <-1
sumstat_pld$pld <-2



### Combining 
combined_list <- rbind(sumstat_crbd, sumstat_bisse, sumstat_ddd, sumstat_pld)
## Defining Training and Testing sets


set.seed(113)
# Define size of datasets.
total_data_points <- n_trees * num_mods
subset_size <- 40000  # Specify the size of the subset

n_train    <- floor(subset_size * .9)
n_valid    <- floor(subset_size * .05)
n_test     <- subset_size - n_train - n_valid
batch_size <- min(subset_size*.01, 64)


# Pick the phylogenies randomly.
ds <- convert_ss_dataframe_to_dataset(combined_list)
# Pick the random subset of data points.
subset_indices <- sample(1:total_data_points, subset_size)

# Split the subset into train, validation, and test indices.
train_indices <- subset_indices[1:n_train]
valid_indices <- subset_indices[(n_train + 1):(n_train + n_valid)]
test_indices  <- subset_indices[(n_train + n_valid + 1):subset_size]


#1 train_indices <- sample(1:nrow(sumstat), n_train)
#1 not_train_indices <- setdiff(1:nrow(sumstat), train_indices)
#1 valid_indices <- sample(not_train_indices, n_valid)
#1 test_indices  <- setdiff(not_train_indices, valid_indices)


target_var<-c("crbd","bisse","ddd","pld")
#target_var<-target_var[-c(3,4)]

# Create the datasets.
train_ds <- ds(combined_list[train_indices, ], target_var, c())
valid_ds <- ds(combined_list[valid_indices, ], target_var, c())
test_ds  <- ds(combined_list[test_indices, ], target_var, c())

# Create the dataloader.
train_dl <- train_ds %>% dataloader(batch_size=batch_size, shuffle=TRUE)
valid_dl <- valid_ds %>% dataloader(batch_size=batch_size, shuffle=FALSE)
test_dl  <- test_ds  %>% dataloader(batch_size=1, shuffle=FALSE)
#test_dl  <- test_ds  %>% dataloader(batch_size=1, shuffle=FALSE)



#### Neural Network Definition(modified for classification problem)
n_in      <- length(train_ds[1]$x) # number of neurons of the input layer 
n_out     <- num_mods # number of classes
n_hidden  <- 100 # number of neurons in the hidden layers 
p_dropout <- 0.01 # dropout probability 
n_epochs  <- 100 # maximum number of epochs for the training 
patience  <- 10 # patience of the early stopping 

# Build the neural network.
dnn.net <- nn_module(
  
  "ss-dnn", 
  
  initialize = function(){
    self$fc1 <- nn_linear(in_features = n_in, out_features = n_hidden)
    self$fc2 <- nn_linear(in_features = n_hidden, out_features = n_hidden)
    self$fc3 <- nn_linear(in_features = n_hidden, out_features = n_hidden)
    self$fc4 <- nn_linear(in_features = n_hidden, out_features = n_hidden)
    self$fc5 <- nn_linear(in_features = n_hidden, out_features = n_out)
  }, 
  
  forward = function(x){
    x %>%
      self$fc1() %>%
      nnf_relu() %>%
      nnf_dropout(p = p_dropout) %>%
      
      self$fc2() %>%
      nnf_relu() %>%
      nnf_dropout(p = p_dropout) %>%
      
      self$fc3() %>%
      nnf_relu() %>%
      nnf_dropout(p = p_dropout) %>%
      
      self$fc4() %>%
      nnf_relu() %>%
      nnf_dropout(p = p_dropout) %>%
      self$fc5()
  }
)

# Set up the neural network.
dnn <- dnn.net() # create CNN
dnn$to(device = device) # Move it to the choosen GPU
opt <- optim_adam(params = dnn$parameters) # optimizer







## Batch functions 
train_batch <- function(b){
  acc<-0
  total<-0
  opt$zero_grad()
  output <- dnn(b$x$to(device = device))
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
  #print("hola2")
  output <- dnn(b$x$to(device = device))
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

#### Training 
epoch     <- 1
trigger   <- 0 
last_loss <- 10


train_losses <- list()
valid_losses <- list()

train_accuracy <- list()
valid_accuracy <- list()



train_plots <- list()
valid_plots <- list()


# Training loop.
start_time <-  Sys.time()
while (epoch < n_epochs & trigger < patience) {
  
  # Training 
  dnn$train()
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
  dnn$eval()
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

print(end_time - start_time)


####Plots


png("loss_curve.png")
# Plot the loss curve
plot(1:length(train_losses), train_losses, type = "l", col = "blue",
     xlab = "Epoch", ylab = "Loss", main = "Training and Validation Loss",
     ylim = range(c(train_losses, valid_losses)))
lines(1:length(valid_losses), valid_losses, type = "l", col = "red")
legend("topright", legend = c("Training Loss", "Validation Loss"),
       col = c("blue", "red"), lty = 1)

# Close the PNG device
dev.off()


png("acc_curve.png")
# Plot the accuracy
plot(1:length(train_accuracy), train_accuracy, type = "l", col = "blue",
     xlab = "Epoch", ylab = "Loss", main = "Training and Validation Accuracy",
     ylim = range(c(train_accuracy, valid_accuracy)))
lines(1:length(valid_accuracy), valid_accuracy, type = "l", col = "red")
legend("topright", legend = c("Training Accuracy", "Validation Accuracy"),
       col = c("blue", "red"), lty = 1)

# Close the PNG device
dev.off()

# Plot the loss curve
#plot(1:length(train_accuracy), train_accuracy, type = "l", col = "blue",
#     xlab = "Epoch", ylab = "Loss", main = "Training and Validation Loss",
#     ylim = range(c(train_accuracy, valid_accuracy)))
#lines(1:length(valid_accuracy), valid_accuracy, type = "l", col = "red")
#legend("topright", legend = c("Training Loss", "Validation Loss"),
#       col = c("blue", "red"), lty = 1)


torch_save(dnn, paste( "models/c01_DNN_1st_try",sep="-"))
cat(paste("\n Model dnn saved", sep = ""))
cat("\nSaving model... Done.")

dnn<-torch_load(paste( "models/c01_DNN_1st_try",sep="-"))


dnn$eval()
pred <- vector(mode = "list", length = n_out)
names(pred)<-target_var

# Computing accuracy on test 

acc_list <- list("crbd"= 0 , "bisse" = 0 ,"ddd" = 0 , "pld"= 0)
total_list <- list("crbd"= 0 , "bisse" = 0 ,"ddd" = 0 , "pld"= 0)



coro::loop(for (b in test_dl) {

  
  
  
})





acc_list <- list("crbd"= 0 , "bisse" = 0 ,"ddd" = 0 , "pld"= 0 ,"total" = 0)
total_list <- list("crbd"= 0 , "bisse" = 0 ,"ddd" = 0 , "pld"= 0, "total"=0)

# Compute accuracy 
coro::loop(for (b in test_dl) {
  output <- dnn(b$x$to(device = device))
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

write.csv(result, file = "Testing_results/dnn.csv", row.names = FALSE)







