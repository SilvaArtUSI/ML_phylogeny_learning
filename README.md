## ML_phylogeny_learning
by Oscar Arturo Silva Castellanos
 Thesis project on classifying  and parameter estimationdifferent phylogeny models
 
# Data Simulaton

For simulating Phylogenetic Trees the code can be found in the folder Rnotebooks_Tree.

Inside this folder the 001_All_trees_simulations.Rmd contains the code for simulating each different kind of trees (CRBD, BiSSE,DDD,LPD)
CRBD and BiSSE can be reescaled to unitary trees with reescalingcrbd_bisse.R


# Neural Network Trainig - Classsifier

1) The simulated data trees need to be saved in the folder data_clas.
2) The Code for training each architecture is in the folder 
3) For the graph training  it is necessary to run the code in python c06_infer_gnn_graph.ipynb in python


# Neural Network Trainig - LDD

1) The simulated data trees need to be saved in the folder data_clas. Im this case only The DDD trees are needed
2) The Code for training each architecture is in the folder RnotebooksDDD

# Neural Network Trainig - LPD

1) The simulated data trees need to be saved in the folder data_clas. Im this case only The DDD trees are needed
2) The Code for training each architecture is in the folder Rnoteboooks_LPD



The starting point for this code comes from the code presented in:
A Comparison of Deep Learning Architectures for Inferring Parameters of Diversification Models from Extant Phylogenies
Ismaël Lajaaiti, Sophia Lambert, Jakub Voznica, Hélène Morlon, Florian Hartig
bioRxiv 2023.03.03.530992; doi: https://doi.org/10.1101/2023.03.03.530992

