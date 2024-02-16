import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
from math import floor


class GNN(torch.nn.Module):
    def __init__(self, num_features_static_graph, gnn_layer, pooling_method, dropout_rate, ratio, hidden_dim, hidden_dim_2, hidden_layers, self_loops):
        super().__init__()
        self.pooling_method = pooling_method
        self.hidden_layers = hidden_layers
        
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout_rate)

        # which gnn layer to use is specified by input argument
        print("Using GAT layers")
        self.conv = gnn.GATConv(
            in_channels=num_features_static_graph, 
            out_channels=hidden_dim, 
            heads=1,
            concat=False,
            add_self_loops=self_loops)

        self.pre_final_linear = nn.Linear(hidden_dim, hidden_dim_2)
        self.final_linear = nn.Linear(hidden_dim_2, 1)
        
        
    def forward(self, x, edge_index, batch=None, target_index=None):
        gnn_out = self.silu(self.conv(x, edge_index))
        for _ in range(self.hidden_layers-1):
            gnn_out = self.silu(self.conv(gnn_out))

        if self.pooling_method=='target':
            out = gnn_out[target_index] 
        elif self.pooling_method=='sum':
            out = gnn.global_add_pool(gnn_out, batch)
        elif self.pooling_method=='mean':
            out = gnn.global_mean_pool(gnn_out, batch)

        out = self.dropout(out)
        out = self.silu(self.pre_final_linear(out))
        # now using BCEwithlogits because more stable, no need of activation function
        out = self.final_linear(out)

        return out
