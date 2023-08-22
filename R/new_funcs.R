###New file for general functions

#### Operations on Rates ####


#' #Takes Compute DDD trees by sampling from the given parameters
#' Formally : 
#' r = lambda - mu
#' epsilon = mu / lambda
#'
#' @param lambda0_c, speciation rate range
#'              mu_c, extinction rate range
#'              carrying_capacity_c,
#'              age1,
#'              ddmodel1
#'              
#'              vector of length 2 containing lambda and mu
#'
#' @return vector of with trees simulation and it's true parameters 
#' @export
#' @examples

generatePhyloDDD <- function(n_trees, lambda0_c,mu_c,carrying_capacity_c,age1,ddmodel1, ss_check = TRUE){
  
  trees <- list()
  extrees <- list()
  Lmats <- list()
  brds_s <- list()
  
  no_NA_ss<-FALSE
  
  name.param <- c("lambda0", "mu","K","age") 
  true.param <- vector(mode='list', length=4)
  names(true.param) <- name.param
  
  while(length(trees) < n_trees){
    
    # Generate phylogeny 
    #tree <- NULL
    #lik  <- NULL
    #while(is.null(tree) | is.null(lik)){
    #  tree <- tree.bisse(vec.param, max.taxa = n_taxa.i, x0 = NA)
    #  if (!(all(tree$tip.state == 0) | all(tree$tip.state == 1))){
    #    lik <- make.bisse(tree, tree$tip.state)
    #  }
    #}
    
    lambda0_sample <- runif(1, min = lambda0_c[1], max = lambda0_c[2])
    #mu_sample <- runif(1, min = mu[1], max = mu[2])
    mu_sample<- runif(1, min = 0, max = lambda0_sample)
    carrying_capacity_sample <- rdunif(1, min = carrying_capacity_c[1], max = carrying_capacity_c[2])
    
    vec.param <- c(lambda0_sample,mu_sample,as.integer(carrying_capacity_sample),as.integer(age1))
    sim.param <- c(lambda0_sample,mu_sample,as.integer(carrying_capacity_sample))
    #print( vec.param)
    
    
    
    outputs <-  dd_sim(pars = sim.param, age = age1, ddmodel = ddmodel1)
    
    tree  <- outputs[[1]]
    extree  <- outputs[[2]]
    Lmat  <- outputs[[3]]
    brds  <- outputs[[4]]
    
    #print(tree$Nnode)
    
    
    
    
    # Checking that summary statistics have no NA
    if (ss_check){
      ss <- get_ss(tree) # compute summary statistics
      no_NA_ss <- !any(is.na(ss)) # does SS have any NA values?
    }
    
    if (no_NA_ss || !ss_check){
      
      trees <- append(trees, list(tree))                    # save tree
      extrees <- append(extrees, list(extree))                    # Additional Battery
      Lmats <- append(Lmats, list(Lmat))                    #
      brds_s <- append(brds_s, list(brds))                    #
      
      
      
      for (i in 1:4){
        
        true.param[[i]] <- c(true.param[[i]], vec.param[i]) # save param.
        
      }
      
      progress(length(trees), n_trees, progress.bar = TRUE, # print
               init = (length(trees)==1))                   # progression
      
    }
  }
  
  out <- list("trees"    = trees, "param"    = true.param, "tas" = extrees, "L" = Lmats,"brts"= brds_s )
  
  return(out)
}

fname_ddd<- function(n_trees, lambda0,mu,carrying_capacity,age,dd_mod){
  
  lambd_text <- ifelse(length(lambda0) == 2, paste(lambda0[1], lambda0[2], sep="-"), 
                       as.character(lambda0))
  mu_text <- ifelse(length(mu) == 2, paste(mu[1], mu[2], sep="-"), 
                    as.character(mu))
  
  k_text <- ifelse(length(carrying_capacity) == 2, paste(carrying_capacity[1], carrying_capacity[2], sep="-"), 
                   as.character(carrying_capacity))
  
  age_text <- as.character(age)
  
  dd_mod_text <- as.character(dd_mod)
  
  
  fname <- paste("nt", n_trees,"la0",lambd_text,"mu",mu_text,"k",k_text,"age",age_text,"ddmod",dd_mod_text ,sep="-")
  
  return(fname)
  
}




# We need to generated new encodings for the DDD
generate_encoding_DDD<- function(phylo_trees,tree_size=1000){
  n_trees <- length(phylo_trees)
  list.encode <- list()
  
  cat("Computing encoding vectors...\n")
  
  for (n in 1:n_trees){
    svMisc::progress(n, n_trees, progress.bar = TRUE, init = (n==1))
    tree <- phylo_trees[[n]] # extract tree
    #print(n)
    tree.encode   <- encode_phylo(tree) # encode the tree
    format.encode <- format_encode(tree.encode, tree_size) # format the encoding
    list.encode[[n]] <- format.encode # save to list 
  }
  
  # Convert the list of vectors to a torch tensor 
  tensor.encode <- as.data.frame(do.call(cbind, list.encode)) %>% 
    as.matrix() 
  
  cat("\nComputing encoding vectors... Done.\n")
  
  return(tensor.encode)
  
}


getmax_nodes<-function(phylo) {
  
  max_nodes<-0
  for (i in seq_along(phylo)) {
  num_nodes <- phylo[[i]]$Nnode  # Get the number of nodes for the current tree
  max_nodes <- max(max_nodes, num_nodes)  # Update the maximum number of nodes if necessary
  }
  max_nodes_rounded <- ceiling(max_nodes / 50) * 50


return (max_nodes_rounded)

}


getmin_nodes<-function(phylo) {
  
  min_nodes<-1000000000
  for (i in seq_along(phylo)) {
    num_nodes <- phylo[[i]]$Nnode  # Get the number of nodes for the current tree
    min_nodes <- min(min_nodes, num_nodes)  # Update the maximum number of nodes if necessary
  }
  
  
  return (min_nodes)
  
}


new_generatePhyloBiSSE<-function(n_trees, n_taxa,param.range, ss_check = TRUE){
  
  no_NA_ss<-FALSE
  trees <- list()
  name.param <- c("lambda0", "lambda1", "mu0", "mu1", "q01", "q10") 
  true.param <- vector(mode='list', length=6)
  names(true.param) <- name.param
  
  lambda.range <- param.range$lambda
  q.range <-param.range$q
  
  
  while(length(trees) < n_trees){
    
    
    lambda0 <-  runif(1, lambda.range[1], lambda.range[2])
    lambda1 <-  runif(1, lambda.range[1], lambda.range[2])
    mu0     <-  runif(1, 0, lambda0 )
    mu1     <-  runif(1, 0, lambda1)
    q01     <-  runif(1, q.range[1], q.range[2])
    q10     <-  runif(1, q.range[1], q.range[2])
    
    vec.param <- c(lambda0, lambda1, mu0, mu1, q01, q10)
    #print(vec.param)
    
    
    
    # draw randomly param. values
    n_taxa.i <- drawPhyloSize(n_taxa)       # draw phylogeny size 
    
    # Generate phylogeny 
    tree <- NULL
    lik  <- NULL
    while(is.null(tree) | is.null(lik)){
      tree <- tree.bisse(vec.param, max.taxa = n_taxa.i, x0 = NA)
      if (!(all(tree$tip.state == 0) | all(tree$tip.state == 1))){
        lik <- make.bisse(tree, tree$tip.state)
      }
    }
    #print(tree$Nnode)
    
    # Checking that summary statistics have no NA
    if (ss_check){
      ss <- get_ss(tree) # compute summary statistics
      no_NA_ss <- !any(is.na(ss)) # does SS have any NA values?
    }
    
    
    if (no_NA_ss || !ss_check){
      trees <- append(trees, list(tree))                    # save tree
      for (i in 1:6){
        true.param[[i]] <- c(true.param[[i]], vec.param[i]) # save param.
      }
      progress(length(trees), n_trees, progress.bar = TRUE, # print
               init = (length(trees)==1))                   # progression
    }
  }
  
  out <- list("trees"    = trees, 
              "param"    = true.param)
  
  return(out)
}


generatePhylo_PLD <- function(n_trees,
                             lambda_interval,
                             K_interval,
                             beta_p_interval,
                             cr_age,
                             DDD=FALSE,
                             ss_check=TRUE,max_tries =10
                             ){
  
  
  trees <- list()
  extrees <- list()
  Lmats <- list()
  brds_s <- list()
  
  name.param <- c("lambda0", "mu","K","beta_p") 
  true.param <- vector(mode='list', length=4)
  names(true.param) <- name.param
  
  j=0
  while(length(trees) < n_trees){
    j=j+1
    
    lambda0_sample <- runif(1, min = lambda_interval[1], max = lambda_interval[2])
    mu_sample <- runif(1, min = 0, max = lambda0_sample)
    k_sample <- runif(1, min = K_interval[1], max = K_interval[2])
    
    if (DDD==TRUE){
      
      beta_p_sample<-0
        
    }else{beta_p_sample <- runif(1, min=beta_p_interval[1], max = beta_p_interval[2] )}
    
    beta_sample = (mu_sample-lambda0_sample)/k_sample
    
    vec.param <- c(lambda0_sample,mu_sample,k_sample,beta_p_sample)
    
    sim.param <- c(mu_sample,lambda0_sample,beta_sample,beta_p_sample)
    #print(sim.param)
    

    outputs <- suppressWarnings(emphasis:::sim_tree_pd_cpp(pars = sim.param,
                                            max_t = 1,
                                            max_lin = 1e+6,
                                            max_tries = max_tries))
  
    
    
    if (outputs$status != "extinct"){
    tree  <- outputs[[1]]
    extree <- outputs[[2]]
    Lmat  <- outputs[[3]]
    brds  <- outputs[[4]]
    
    trees <- append(trees, list(tree))                    # save tree
    extrees <- append(extrees, list(extree))                    # Additional Battery
    Lmats <- append(Lmats, list(Lmat))                    #
    brds_s <- append(brds_s, list(brds))                    #
    
    for (i in 1:4){
      true.param[[i]] <- c(true.param[[i]], vec.param[i]) # save param.
    }
    
    }
    #print(length(trees))
    
    svMisc:::progress(length(trees), n_trees, progress.bar = TRUE, # print
                      init = (length(trees)==1))                   # progression
    
  }
  
  out <- list("trees"    = trees, "param"    = true.param, "tas" = extrees, "L" = Lmats,"brts"= brds_s )
  
  return(out)
}

calculate_accuracy <- function(predicted, target) {
  correct_predictions <- sum(predicted == target)
  total_predictions <- length(target)
  accuracy <- correct_predictions / total_predictions
  return(accuracy)
}



rescale_tree <- function(phy, new_crown_age = 1) {
  ltable <- DDD::phylo2L(phy)
  ltable[, 1] <- new_crown_age * ltable[, 1] / max(ltable[, 1])
  new_phy <- DDD::L2phylo(ltable)
  return(new_phy)
}


min_max_normalize<-function(x,rang=1){
  
  if (length(rang)==1){ var<-(x -min(x))/(max(x)-min(x)) }
  else{
    
    var<-(x-rang[1])/(rang[2]-rang[1])
    
  }
  
  return(var)
  
}

min_max_unnormalize<-function(x,rang){
  
  var <- x*(rang[2]-rang[1])+rang[1]
  
  return (var)
  
}
  
  
  

