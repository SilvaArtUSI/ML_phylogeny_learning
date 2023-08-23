



new_generate_ltt_dataframe<-
  function(trees, n_taxa){
    
    n_trees  <- length(trees) # number of trees 
    n_row <- ifelse(length(n_taxa) == 1, n_taxa, n_taxa[2])
    df.ltt <- data.frame("tree1" = rep(NA, n_row))
    
    #df.rates <- as.data.frame(do.call(cbind, true.param))
    
    cat("Creating LTT dataframe...\n")
    
    for (i in 1:n_trees){
      tree <- trees[[i]] # get tree 
      ltt.coord <- ape::ltt.plot.coords(tree) # get ltt coordinates 
      ltt.coord <- as.data.frame(ltt.coord)
      ltt.coord.time <- ltt.coord$time
      n <- length(ltt.coord.time)
      df.ltt[1:n,paste("tree", i, sep = "")] <- ltt.coord$time
      progress(i, n_trees, progress.bar = TRUE, init = (i==1))
    }
    
    cat("\nCreating LTT dataframe... Done.")
    #out <- list("ltt" = df.ltt, "rates" = df.rates)
    out <- df.ltt # function output
    
    return(out)
    
  }

new_convert_ltt_dataframe_to_dataset<-function(df.ltt, nn_type){
  
  if (nn_type == "cnn-ltt"){
    ds.ltt <- torch::dataset(
      name <- "ltt_dataset", 
      initialize = function(df.ltt){
        
        # input
        df.ltt[is.na(df.ltt)] <- 0
        
        array.ltt <- df.ltt %>% 
          as.matrix() %>% 
          torch_tensor()
        self$x <- array.ltt
        
        # target 
        # Remove the following line as it references true.param
        # self$y <- torch_tensor(do.call(cbind, true.param)) # target
      }, 
      
      .getitem = function(i) {list(x = self$x[,i]$unsqueeze(1))},
      
      .length = function() {self$x$size()[[2]]}
    )
  }
  
  else{
    ds.ltt <- torch::dataset(
      name <- "ltt_dataset", 
      initialize = function(df.ltt){
        
        # input
        df.ltt[is.na(df.ltt)] <- 0
        
        array.ltt <- df.ltt %>% 
          as.matrix() %>% 
          torch_tensor()
        self$x <- array.ltt
        
        # target 
        # Remove the following line as it references true.param
        # self$y <- torch_tensor(do.call(cbind, true.param)) # target
      }, 
      
      .getitem = function(i) {list(x = self$x[,i])},
      
      .length = function() {self$x$size()[[2]]}
    )
  }
  
  return(ds.ltt)
}



#Loading phylo

phylo <- readRDS( "data_clas/phylogeny-ddd-nt-10000-la0-0.5-1.5-mu-0.05-0.5-k-10-100-age-1-ddmod-1.rds")

max_nodes_rounded<-150 #size of the input when trained

#transforming data
df.ltt_new<-new_generate_ltt_dataframe(phylo,max_nodes_rounded)
#ds.ltt <- convert_ltt_dataframe_to_dataset(df.ltt_test, out_phyloddd$param, "cnn-ltt")
ds.ltt_new <- new_convert_ltt_dataframe_to_dataset(df.ltt_test,  "cnn-ltt")

#test_ds  <- ds.ltt(df.ltt[, test_indices] , extract_elements(true, test_indices))
ds_eval  <- ds.ltt_new(df.ltt_new)


data_loader_ltt <- ds_eval  %>% dataloader(batch_size=1,     shuffle=FALSE)



device<-"cpu"
cnn_ltt<-torch_load( 'M04_CNNLTT-DDD-K-40-100-10000-Lay-3-Hn-8-ker-5-p-10')
cnn_ltt$eval()
n_out<-3
nn.pred <- vector(mode = "list", length = n_out)
names(nn.pred) <- names(c("lambda0", "mu"    ,  "K" ))


coro::loop(for (b in data_loader_ltt) {
  #if (model_type == "crbd"){b$x <- b$x$unsqueeze(2)}
  #print(b$x)
  out <- cnn_ltt(b$x$to(device = device))
  pred <- as.numeric(out$to(device = "cpu")) # move the tensor to CPU 
  #true <- as.numeric(b$y)
  for (i in 1:n_out){nn.pred[[i]] <- c(nn.pred[[i]], pred[i])}
})
