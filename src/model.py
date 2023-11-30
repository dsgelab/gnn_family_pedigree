import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
from math import floor


class GNN(torch.nn.Module):
    def __init__(self, num_features_static_graph, gnn_layer, pooling_method, dropout_rate, ratio, hidden_dim, hidden_layers):
        super().__init__()
        self.pooling_method = pooling_method
        self.hidden_layers = hidden_layers
        half_hidden_dim = floor(hidden_dim/2)

        # which gnn layer to use is specified by input argument
        if gnn_layer=='gcn':
            print("Using GCN layers")
            self.conv1 = gnn.GCNConv(num_features_static_graph, hidden_dim, add_self_loops=False)
            self.conv2 = gnn.GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
        if gnn_layer=='graphconv':
            print("Using GraphConv layers")
            self.conv1 = gnn.GraphConv(num_features_static_graph, hidden_dim, add_self_loops=False)
            self.conv2 = gnn.GraphConv(hidden_dim, hidden_dim, add_self_loops=False)
        elif gnn_layer=='gat':
            print("Using GAT layers")
            self.conv1 = gnn.GATConv(num_features_static_graph, hidden_dim, add_self_loops=False)
            self.conv2 = gnn.GATConv(hidden_dim, hidden_dim, add_self_loops=False)

        self.pre_final_linear = nn.Linear(hidden_dim,half_hidden_dim)
        self.final_linear = nn.Linear(half_hidden_dim, 1)
        
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, edge_index, edge_weight=None, batch=None, target_index=None):

        gnn_out = self.silu(self.conv1(x, edge_index, edge_weight))
        for i in range(self.hidden_layers):
            gnn_out = self.silu(self.conv2(gnn_out, edge_index, edge_weight))
        
        if self.pooling_method=='target':
            out = gnn_out[target_index] 
        elif self.pooling_method=='sum':
            out = gnn.global_add_pool(gnn_out, batch)
        elif self.pooling_method=='mean':
            out = gnn.global_mean_pool(gnn_out, batch)
            
        out = self.dropout(out)  
        out = self.silu(self.pre_final_linear(out))     
        #out = self.sigmoid(self.final_linear(out))

        # now using BCEwithlogits because more stable
        out = self.final_linear(out)

        return out
