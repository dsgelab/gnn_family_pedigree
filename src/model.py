import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F


class GNN(torch.nn.Module):
    def __init__(self, num_features_static_graph, hidden_dim, gnn_layer, pooling_method, dropout_rate, ratio, loss):
        super().__init__()
        self.pooling_method = pooling_method
        self.loss = loss

        # which gnn layer to use is specified by input argument
        if gnn_layer=='gcn':
            print("Using GCN layers")
            self.conv1 = gnn.GCNConv(num_features_static_graph, hidden_dim)
            self.conv2 = gnn.GCNConv(hidden_dim, hidden_dim)
        if gnn_layer=='graphconv':
            print("Using GraphConv layers")
            self.conv1 = gnn.GraphConv(num_features_static_graph, hidden_dim)
            self.conv2 = gnn.GraphConv(hidden_dim, hidden_dim)
        elif gnn_layer=='gat':
            print("Using GAT layers")
            self.conv1 = gnn.GATConv(num_features_static_graph, hidden_dim)
            self.conv2 = gnn.GATConv(hidden_dim, hidden_dim)

        self.pre_final_linear = nn.Linear(hidden_dim,hidden_dim)
        self.final_linear = nn.Linear(hidden_dim, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x_static_graph, edge_index, edge_weight, batch, target_index):

        gnn_out = self.relu(self.conv1(x_static_graph, edge_index, edge_weight))
        gnn_out = self.relu(self.conv2(gnn_out, edge_index, edge_weight))
        
        if self.pooling_method=='target':
            out = gnn_out[target_index] 
        elif self.pooling_method=='sum':
            out = gnn.global_add_pool(gnn_out, batch)
        elif self.pooling_method=='mean':
            out = gnn.global_mean_pool(gnn_out, batch)
            
        out = self.dropout(out)  
        out = self.relu(self.pre_final_linear(out))
        if self.loss=='bce':
            out = self.sigmoid(self.final_linear(out))
        else:
            out = self.relu(self.final_linear(out))

        return out
