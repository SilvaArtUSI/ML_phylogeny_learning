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

