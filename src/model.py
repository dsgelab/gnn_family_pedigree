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
        
        self.TopKpool = gnn.TopKPooling(hidden_dim, ratio=ratio)
        self.SAGpool = gnn.SAGPooling(hidden_dim, ratio=ratio)

        # which gnn layer to use is specified by input argument
        if gnn_layer=='gat':
            print("Using GAT layers")
            self.conv = gnn.GATConv(
                in_channels=num_features_static_graph, 
                out_channels=hidden_dim, 
                num_layers=self.hidden_layers, 
                heads=1,
                dropout=dropout_rate,
                add_self_loops=self_loops)

        self.final_linear = nn.Linear(hidden_dim, 1)
        
        
    def forward(self, x, edge_index, batch=None, target_index=None):
        gnn_out = self.silu(self.conv(x, edge_index))

        if self.pooling_method=='target':
            out = gnn_out[target_index] 
        elif self.pooling_method=='sum':
            out = gnn.global_add_pool(gnn_out, batch)
        elif self.pooling_method=='mean':
            out = gnn.global_mean_pool(gnn_out, batch)
        elif self.pooling_method=='topkpool_sum':
            out, _, _, pool_batch, _, _ = self.TopKpool(gnn_out, edge_index, batch)
            out = gnn.global_add_pool(out, pool_batch)
        elif self.pooling_method=='topkpool_mean':
            out, _, _, pool_batch, _, _ = self.TopKpool(gnn_out, edge_index, batch)
            out = gnn.global_mean_pool(out, pool_batch)
        elif self.pooling_method=='sagpool_sum':
            out, _, _, pool_batch, _, _ = self.SAGpool(gnn_out, edge_index, batch)
            out = gnn.global_add_pool(out, pool_batch)
        elif self.pooling_method=='sagpool_mean':
            out, _, _, pool_batch, _, _ = self.SAGpool(gnn_out, edge_index, batch)
            out = gnn.global_mean_pool(out, pool_batch)  

        # now using BCEwithlogits because more stable, no need of activation function
        out = self.final_linear(out)

        return out
