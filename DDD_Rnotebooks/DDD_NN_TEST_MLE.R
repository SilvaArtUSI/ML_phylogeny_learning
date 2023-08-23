

n_hidden_cnn <- 8
n_layer_cnn  <- 4
ker_size_cnn <- 10
n_input_cnn  <- nrow(cblv)
n_out    <- 3
p_dropout <- 0


# Build the CNN

cnn.net2 <- nn_module(
  
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

cnn <- cnn.net2(n_input_cnn, n_out, n_hidden_cnn, n_layer_cnn, ker_size_cnn) # create CNN
cnn$to(device = device) # Move it to the choosen GPU
cnn<-torch_load( 'data_DDD/models_DDD/04_CNN-lay-4-hid-8')



device<-"cpu"
n_hidden_cnnltt  <- 16
n_layer_cnnltt   <- 3
ker_size_cnnltt  <- 5
p_dropout_cnnltt <- 0
n_input_cnnltt   <- max_nodes_roundedcnnltt
n_out     <- 3



# Build the CNN

cnn.net1 <- nn_module(
  
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
cnn_ltt <- cnn.net1(n_input_cnnltt, n_out, n_hidden_cnnltt, n_layer_cnnltt, ker_size_cnnltt, p_dropout_cnnltt) # create CNN
cnn_ltt$to(device = device) # Move it to the choosen GPU
cnn_ltt<-torch_load( 'data_DDD/models_DDD/04_CNNLTT lay 3 hid -')
cnn_ltt$eval()

