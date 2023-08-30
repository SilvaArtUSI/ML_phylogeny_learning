# Install and load the ape library if not already installed
# install.packages("ape")
library(ape)


foo <- function() {
   col <- "black"
  for (i in 1:2)
     axis(i, col = col, col.ticks = col, col.axis = col, las = 1)
  box(lty = "19")
   }

# Create a simple phylogenetic tree with 5 species
tree <- read.tree(text = "(A:3,(B:2,(C:0.3,(D:2,E:2):0.6):0.7):0.8);")
png("phylogenetic_tree.png")
total_height <- max(node.height(tree.owls))
plot(tree.owls, show.tip.label = TRUE, cex = 0.8, edge.width = 2,
     x.lim = c(0, total_height),res=300)
axisPhylo(backward = FALSE)
dev.off()

phylo.plot(phylo[[3]])

