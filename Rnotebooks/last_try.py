import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GCNConv, SAGEConv # Graph Neural Network 
from torch_geometric.nn import global_mean_pool 
import rpy2.robjects as robjects # load R object 
from rpy2.robjects import pandas2ri # load R object 
from tqdm import tqdm # print progress bar 
import pickle # save object 
import matplotlib.pyplot as plt
import numpy as np
import random 
import pandas as pd
import os



path = r"/mnt/c/Users/oasc_/Documents/Thesis/ML_phylogeny_learning"
os.getcwd()
os.chdir(path )
print(os.getcwd())


pandas2ri.activate()

#fname_graph = "data/phylogeny-DDD-nt-1e+05-la0-0.5-1.5-mu-0.05-0.5-k-10-20-age-15-ddmod-1-graph.rds"
#fname_param = "data/true-param-DDD-nt-1e+05-la0-0.5-1.5-mu-0.05-0.5-k-10-20-age-15-ddmod-1.rds"


fname_graph =  path + r"/data_clas/phylogeny-all-graph.rds"
fname_param = path + r"/data_clas/true-param-all-graph.rds"

readRDS = robjects.r['readRDS']
df_graph = readRDS(fname_graph)
df_graph = pandas2ri.rpy2py(df_graph)
df_param = readRDS(fname_param)
df_param = pandas2ri.rpy2py(df_param)


#df_param[2] = df_param[2]/100
#df_param[3] = df_param[3]/100

#removing crown age from predictions
#df_param = df_param[0:3]


preselected_sets = False
file_of_index = "data/10_3_indices_set_-DDD-Totalset-10000-SubSiz-10000-.rds" # to use the same as R

if preselected_sets == True:
    ind = readRDS(file_of_index) 
    ind = pandas2ri.rpy2py(ind)

    train, valid, test = ind
    train_ind, valid_ind, test_ind = list(train), list(valid), list(test)
    #print(train_ind[1])
    train_ind, valid_ind, test_ind = [i-1 for i in train_ind], [i-1 for i in valid_ind], [i-1 for i in test_ind]
    #print(train_ind[1])

else:
    #total_data_points = len(df_graph)
    total_data_points = 40000
    subset_size = 40000-1 # Specify the size of the subset

    n_train = int(subset_size * 0.9)
    n_valid = int(subset_size * 0.05)
    n_test = subset_size - n_train - n_valid
    batch_size = min(int(subset_size * 0.01), 10)

    # Pick the phylogenies randomly.
    #true[2] = true[2] / 100
    #true[3] = true[3] / 100
    #ds = convert_encode_to_dataset(cblv, true)

    # Pick the random subset of data points.
    subset_indices = random.sample(range(1, total_data_points), subset_size)

    # Split the subset into train, validation, and test indices.
    train_ind = subset_indices[:n_train]
    valid_ind = subset_indices[n_train:(n_train + n_valid)]
    test_ind = subset_indices[(n_train + n_valid):subset_size]

def convert_df_to_tensor(df_node, df_edge, params):

    """
    Convert the data frames containing node and edge information 
    to a torch tensor that can be used to feed neural 
    """

    n_node, n_edge = df_node.shape[0], df_edge.shape[0]
    #n_node, n_edge = int(robjects.r("nrow")(df_node)[0]), int(robjects.r("nrow")(df_edge)[0])

    #pandas_df_node = pd.DataFrame(df_node)
    #pandas_df_edge = pandas2ri.ri2py_dataframe(df_edge)

    l1, l2 = [], []
    
    for i in range(n_edge):
        #print(type(pandas_df_node))
        edge = df_edge.iloc[i]
        u, v = edge[0]-1, edge[1]-1
        l1 = l1 + [u,v]
        l2 = l2 + [v,u]

    edge_index = torch.tensor([l1,l2], dtype=torch.long)

    x = []

    for i in range(n_node):
        node_attr = list(df_node.iloc[i])
        x.append(node_attr)

    x = torch.tensor(x, dtype = torch.float)

    y = torch.tensor(params, dtype = torch.float)

    data = Data(x = x, edge_index = edge_index, y = y)

    return(data)


save = False

fname = fname_param[:-9] + "geomtensor" + ".obj" # file name 
if (save):
    print("Save")
    file = open(fname, "wb") # file handler 
    pickle.dump(data_list, file) # save data_list

else:
    print("Load")
    file = open(fname, "rb")
    data_list = pickle.load(file) 

del df_graph
del df_param


#device = "cuda:2" # GPU to use 
device="cuda"
batch_size_max = 32


train_data = [data_list[i].to(device=device) for i in train_ind]
valid_data = [data_list[i].to(device=device) for i in valid_ind]
test_data  = [data_list[i].to(device=device) for i in test_ind]

train_dl = DataLoader(train_data, batch_size = batch_size_max, shuffle = True)
valid_dl = DataLoader(valid_data, batch_size = batch_size_max, shuffle = True)
test_dl  = DataLoader(test_data , batch_size = 1)



class GCN(torch.nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.conv1 = GCNConv(n_in, n_hidden)
        self.conv2 = GCNConv(n_hidden, n_hidden)
        self.conv3 = GCNConv(n_hidden, n_hidden)
        self.conv4 = GCNConv(n_hidden, n_hidden)
        self.conv5 = GCNConv(n_hidden, 2*n_hidden)
        self.lin1  = torch.nn.Linear(2*n_hidden, n_hidden)
        self.lin2  = torch.nn.Linear(n_hidden, n_out)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p = 0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p = 0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p = 0.5, training=self.training)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p = 0.5, training=self.training)
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p = 0.5, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.lin1(x)
        x = self.lin2(x)
        return x
    
def train(model, batch):
    optimizer.zero_grad()
    out = model(batch)
    batch_size = int(max(data.batch) + 1) # number of trees in the batch 
    loss = F.cross_entropy(out, data.y.reshape([batch_size, n_out])) # compute loss 
    loss.backward() # backward propagation 
    optimizer.step()
    acc=torch.sum(torch.argmax(data.y.reshape([batch_size, n_out]),axis=1)==torch.argmax(out,dim=1)).item()
    return(loss,acc/batch_size)

def valid(model, batch):
    out = model(batch)
    batch_size = int(max(data.batch) + 1) # number of trees in the batch 
    loss = F.cross_entropy(out, data.y.reshape([batch_size, n_out])) # compute loss
    acc=torch.sum(torch.argmax(data.y.reshape([batch_size, n_out]),axis=1)==torch.argmax(out,dim=1)).item()
    #total = out.size(dim=0)
    return(loss,acc/batch_size)

# Setting up the training 
n_in = data_list[0].num_node_features
n_out = len(data_list[0].y)
n_hidden = 100
n_epochs = 100
model = GCN(n_in, n_hidden, n_out).to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss_fn = F.cross_entropy


valid_list = []
train_list = []

train_acc_list = []
valid_acc_list = []


patience = 10  # Number of epochs to wait for improvement
best_loss = float('inf')
best_epoch = 0

early_stop = False
counter = 0



# Training loop 

for epoch in range(n_epochs):
    if early_stop:
        break

    # Training
    model.train()
    train_loss = []
    train_acc = []
    for data in tqdm(train_dl):
        loss , acc = train(model, data)  # train model and get loss
        loss = float(loss.to(device="cpu"))
        train_loss.append(loss)
        train_acc.append(acc)
    mean_loss = np.mean(train_loss)
    mean_acc = np.mean(train_acc)

    train_list.append(mean_loss)
    train_acc_list.append(mean_acc)

    print("Epoch %d - Train Loss : %.4f Train Acc : %.4f" % (epoch, float(mean_loss),float(mean_acc)))  # print progression

    # Validation
    model.eval()
    valid_loss = []
    valid_acc = []
    for data in tqdm(valid_dl):
        loss,acc = valid(model, data)  # train model and get loss
        loss = float(loss.to(device="cpu"))
        valid_loss.append(loss)
        valid_acc.append(acc)
    mean_loss = np.mean(valid_loss)
    mean_acc = np.mean(valid_acc)


    valid_list.append(mean_loss)
    valid_acc_list.append(mean_acc)
    print("Epoch %d - Valid Loss : %.4f Train Acc : %.4f" % (epoch, float(mean_loss),float(mean_acc)))  # print progression

    # Check for early stopping
    if mean_loss < best_loss:
        best_loss = mean_loss
        best_epoch = epoch
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping at epoch %d" % (epoch))
            early_stop = True

print("Best epoch:", best_epoch)

# Specify the file path to save the model
file_path = 'GraphNeural.pt'

# Save the model
torch.save(model.state_dict(), file_path)