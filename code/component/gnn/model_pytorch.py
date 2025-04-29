# Description:
# Defines a Heterogeneous Graph Neural Network (HeteroGNN) model using 
# PyTorch Geometric's HeteroConv module and SAGEConv for message passing.
#
# Components:
# - input_proj : Projects raw input features of each node type to a shared hidden size.
# - layers     : A stack of HeteroConv layers for heterogeneous message passing.
# - lin        : Final linear layer to produce classification logits for the target node type.
#
# Class:
# - HeteroGNN:
#   - __init__: Initializes projection layers, HeteroConv layers, and final classifier.
#   - forward: Performs message passing across heterogeneous node and edge types.
#
# Usage:
# Instantiate the HeteroGNN class with appropriate input dimensions and pass 
# a HeteroData object (from PyTorch Geometric) through it to obtain output logits.
# Suitable for node-level classification tasks on heterogeneous graphs.
# ------------------------------------------------------------------------------


from component.packages import *

class HeteroGNN(torch.nn.Module):
    def __init__(self, in_size_dict, hidden_size, out_size, n_layers, etypes, target_node):
        """
        Defines a Heterogeneous Graph Neural Network using PyTorch Geometric.
        """
        super().__init__()
        self.target_node = target_node

        # Input projection layers to ensure all node types have hidden_size embeddings
        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(in_size_dict[ntype], hidden_size) for ntype in in_size_dict
        })

        # Define HeteroConv layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(HeteroConv({
                etype: SAGEConv((-1, -1), hidden_size) for etype in etypes
            }, aggr='sum'))

        # Final linear layer for classification
        self.lin = Linear(hidden_size, out_size)

    def forward(self, data):
        """
        Forward pass of the Heterogeneous GNN model.
        """
        # Apply input projection to make all node features of size hidden_size
        x_dict = {ntype: self.input_proj[ntype](data.x_dict[ntype]) for ntype in data.x_dict}

        # Apply HeteroConv layers
        for i, layer in enumerate(self.layers):
            x_dict = layer(x_dict, data.edge_index_dict)
            if i != len(self.layers) - 1:  # Apply activation except for last layer
                x_dict = {k: F.leaky_relu(x) for k, x in x_dict.items()}

        return self.lin(x_dict[self.target_node])  # Predict target node embeddings
