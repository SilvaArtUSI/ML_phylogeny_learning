

library(torch)
library(luz)
source("R/infer-general-functions.R")
source("R/neural-network-functions.R")
source("R/convert-phylo-to-cblv.R")
source("R/new_funcs.R")



# Importing libraries



# Set parameters DDD model simulation
## Factor for unitary trees 15
n_trees <- 10000# number of trees to generate

dd_model = 1 # dd_model = 1 linear dependence in speciation rate with parameter K (= diversity where speciation = extinction)

device <- "cpu" # GPU where to run computations 
nn_type <- "rnn-ltt" # type of the model: Recurrent Neural Network w/ LTT



max_nodes_rounded<-1200
min_nodes_rounded<-1
n_mods<-4
generateltt<-FALSE






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



set.seed(113)
total_data_points<-n_trees*n_mods
subset_size <- n_trees*n_mods # Specify the size of the subset

n_train    <- floor(subset_size * .9)
n_valid    <- floor(subset_size * .05)
n_test     <- subset_size - n_train - n_valid
batch_size <- 64
patience   <- 10


ds.ltt <- convert_ltt_dataframe_to_dataset(df.ltt, true, nn_type)


# Pick the random subset of data points.
subset_indices <- sample(1:total_data_points, subset_size)

# Split the subset into train, validation, and test indices.
train_indices <- subset_indices[1:n_train]
valid_indices <- subset_indices[(n_train + 1):(n_train + n_valid)]
test_indices  <- subset_indices[(n_train + n_valid + 1):subset_size]



n_taxa=c(1,max_nodes_rounded)



# Creation of the train, valid and test dataset

if (length(n_taxa) == 1){
  
  # Creation of the dataset 
  train_ds <- ds.ltt(df.ltt[, train_indices], extract_elements(true, train_indices))
  valid_ds <- ds.ltt(df.ltt[, valid_indices], extract_elements(true, valid_indices))
  test_ds  <- ds.ltt(df.ltt[, test_indices] , extract_elements(true, test_indices))
  
  # Creation of the dataloader 
  train_dl <- train_ds %>% dataloader(batch_size=batch_size, shuffle=TRUE)
  valid_dl <- valid_ds %>% dataloader(batch_size=batch_size, shuffle=FALSE)
  test_dl  <- test_ds  %>% dataloader(batch_size=1,          shuffle=FALSE)
}

if (length(n_taxa) == 2){
  
  # Training set 
  true.param.train <- true %>% as.data.frame()
  true.param.train <- true.param.train[train_indices, ] %>% as.list()
  list.indices.train <- find_indices_same_size(df.ltt[, train_indices], n_taxa)
  train.set   <- create_all_batch(df.ltt[, train_indices],
                                  true.param.train, list.indices.train,
                                  n_taxa)
  train.set <- reformat_set(train.set, max_batch_size = batch_size)
  
  # Validation set 
  true.param.valid <- true %>% as.data.frame()
  true.param.valid <- true.param.valid[valid_indices, ] %>% as.list()
  list.indices.valid <- find_indices_same_size(df.ltt[, valid_indices], n_taxa)
  valid.set   <- create_all_batch(df.ltt[, valid_indices],
                                  true.param.valid, list.indices.valid,
                                  n_taxa)
  valid.set <- reformat_set(valid.set, max_batch_size = batch_size)
  
  # Testing set 
  true.param.test <- true %>% as.data.frame()
  true.param.test <- true.param.test[test_indices, ] %>% as.list()
  list.indices.test <- list()
  for (i in 1:n_test){list.indices.test[[i]] = i}
  #list.indices.test <- find_indices_same_size(df.ltt[, test_indices], n_taxa)
  test.set   <- create_all_batch(df.ltt[, test_indices],
                                 true.param.test, list.indices.test,
                                 n_taxa)
  test.set <- reformat_set(test.set, max_batch_size = 1)
}




# Parameters of the RNN
n_hidden  <- 100   # number of neurons in hidden layers 
n_layer   <- 2  # number of stacked RNN layers 
p_dropout <- .01 # dropout probability
n_out     <- length(true)


# Build the RNN 
rnn.net <- nn_module(
  initialize = function(n_input, n_out, n_hidden, n_layer, p_dropout = .01,
                        batch_first = TRUE) {
    self$rnn <- nn_lstm(input_size = n_input, hidden_size = n_hidden, 
                        dropout = p_dropout, num_layers = n_layer,
                        batch_first = batch_first)
    self$out <- nn_linear(n_hidden, n_out)
  },
  
  forward = function(x) {
    x <- self$rnn(x)[[1]]
    x <- x[, dim(x)[2], ]
    x %>% self$out() 
  }
)

rnn <- rnn.net(1, n_out, n_hidden, n_layer, p_dropout) # create the RNN
rnn$to(device = device) # move the RNN to the choosen GPU 
opt <- optim_adam(params = rnn$parameters) # optimizer 


# Prepare training 

train_batch <- function(b){
  opt$zero_grad()
  #if (model_type == "crbd"){b$x <- b$x$unsqueeze(2)}
  b$x <- b$x$squeeze(2)$unsqueeze(3)
  output <- rnn(b$x$to(device = device))
  target <- b$y$to(device = device)
  loss <- nnf_cross_entropy(output, target)
  loss$backward()
  opt$step()
  loss$item()
  
  # Compute accuracy
  max_indices1 <- torch_argmax(output, dim = 2,keepdim =FALSE)  # Find the predicted class labels
  max_indices2 <- torch_argmax(target, dim = 2, keepdim =FALSE)  # Find the true class labels
  
  #print(max_indices1)
  
  acc <- torch_sum(max_indices1 == max_indices2)
  total <- length(max_indices1)
  
  return(list(loss = loss$item(), accuracy = acc$item(), total = total))
  
}

valid_batch <- function(b) {
  #if (model_type == "crbd"){b$x <- b$x$unsqueeze(2)}
  b$x <- b$x$squeeze(2)$unsqueeze(3)
  output <- rnn(b$x$to(device = device))
  target <- b$y$to(device = device)
  loss <- nnf_cross_entropy(output, target)
  loss$item()
  
  # Compute accuracy
  max_indices1 <- torch_argmax(output, dim = 2,keepdim =FALSE)  # Find the predicted class labels
  max_indices2 <- torch_argmax(target, dim = 2, keepdim =FALSE)  # Find the true class labels
  
  #print(max_indices1)
  
  acc <- torch_sum(max_indices1 == max_indices2)
  total <- length(max_indices1)
  
  return(list(loss = loss$item(), accuracy = acc$item(), total = total))
  
}


# Training loop


epoch  <- 1
n_epochs<-100
trigger   <- 0 

last_loss <- 100



train_losses <- list()
valid_losses <- list()

train_accuracy <- list()
valid_accuracy <- list()


train_plots <- list()
valid_plots <- list()

start_time <- Sys.time()

while (epoch <= n_epochs & trigger < patience) {
  
  # Training part 
  rnn$train()
  train_loss <- c()
  train_accu <- c()
  

    n_train_batch <- length(train.set)
    n_valid_batch <- length(valid.set)
    random_iter <- sample(1:n_train_batch, n_train_batch)
    c <- 0
    coro::loop(for (i in random_iter) {
      b <- train.set[[i]]
      #print(dim(b$x))
      loss <- train_batch(b)
      train_loss <- c(train_loss, loss$loss)
      train_accu <- c(train_accu, loss$accuracy/loss$total)
      
      c <- c + 1
      #print(c)
    })
  
  
  mean_tl<-mean(train_loss)
  mean_ta<-mean(train_accu)
  
  # Print Epoch and value of Loss function 
  cat(sprintf("epoch %0.3d/%0.3d - train - loss: %3.5f - accuracy: %3.5f \n",
              epoch, n_epochs, mean_tl, mean_ta))
  
  

  
  # Evaluation part 
  rnn$eval()
  valid_loss <- c()
  valid_accu <- c()
  

    coro::loop(for (i in 1:n_valid_batch) {
      b <- valid.set[[i]]
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
  
  epoch <- epoch + 1 
  train_losses <- c(train_losses, mean(train_loss))
  valid_losses <- c(valid_losses, current_loss)
  train_accuracy <-c(train_accuracy,mean_ta)
  valid_accuracy <-c(valid_accuracy,current_accu)
}

end_time <- Sys.time()

run_timelstm <- end_time - start_time
print(run_timelstm)


# Plot the loss curve
plot(1:length(train_losses), train_losses, type = "l", col = "blue",
     xlab = "Epoch", ylab = "Loss", main = "Training and Validation Loss",
     ylim = range(c(train_losses, valid_losses)))
lines(1:length(valid_losses), valid_losses, type = "l", col = "red")
legend("topright", legend = c("Training Loss", "Validation Loss"),
       col = c("blue", "red"), lty = 1)


torch_save(rnn, paste( "M05_LSTM-DDD-",subset_size,"Lay",n_layer,"Hn",n_hidden,"p",patience,sep="-"))
cat(paste("\n Model cnn ltt saved", sep = ""))
cat("\nSaving model... Done.")

rnn<-torch_load( paste( "M06_LSTM-DDD-",subset_size,"Lay",n_layer,"Hn",n_hidden,"p",patience,sep="-"))

rnn<-torch_load( paste( "M05_LSTM-DDD-K",k[1],k[2],subset_size,"Lay",n_layer,"Hn",n_hidden,"p",patience,sep="-"))
rnn$to(device =device  )



# Evaluation of the predictions of the RNN w/ test set

```{R}
rnn$eval() 
rm(nn.pred)
nn.pred <- vector(mode = "list", length = n_out)
names(nn.pred) <- names(true)

if (length(n_taxa) == 2){
  coro::loop(for (i in 1:n_test) {
    b   <- test.set[[i]]
    out <- rnn(b$x$unsqueeze(3)$to(device = device))
    out <- out$squeeze(1)$to(device = "cpu") %>% as.numeric()
    #true <- b$y %>% as.array()
    for (i in 1:length(out)){
      nn.pred[[i]] <- c(nn.pred[[i]],out[i])
      #test.true[[i]] <- c(test.true[[i]],true[i])
    }
  })
}

if (length(n_taxa) == 1) {
  coro::loop(for (b in test_dl) {
    out <- rnn(b$x$reshape(c(b$x$shape, 1L))$to(device = device))
    pred <- as.numeric(out$to(device = "cpu")) # move the tensor to CPU 
    true <- as.numeric(b$y)
    vec.pred.lambda <- c(vec.pred.lambda, pred[1])
    vec.pred.mu     <- c(vec.pred.mu, pred[2])
    vec.true.lambda <- c(vec.true.lambda, true[1])
    vec.true.mu     <- c(vec.true.mu, true[2])
  })
}



    
    

# Example list of lists


# Convert list to data frame
#my_df <- data.frame(matrix(unlist(nn.pred), nrow = length(nn.pred), byrow = TRUE))

# Export data frame to CSV
#write.csv(my_df, "pred.csv", row.names = FALSE)


# Create a vector to store the phylo$Node values
node_values_ddd <- c()
node_values_crbd <- c()
node_values_bisse  <- c()

# Iterate over each tree in the list
for (i in seq_along(phylo)) {
  num_nodes <- phylo[[i]]$Nnode  # Get the Node values for the current tree
  node_values <- c(node_values, num_nodes)  # Append the Node values to the vector
}

# Plot a histogram of node_values
hist(node_values, breaks = "FD", main = "Histogram of Node Values", xlab = "Node Values")
