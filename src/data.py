import torch
from torch.utils.data import Dataset, sampler

import torch_geometric
from torch_geometric.data import Dataset as GraphDataset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

import pandas as pd
import numpy as np
from random import choices
import multiprocessing


GRAPH_NODE_STRUCTURE = {
  # 0:'target',
  1:'father',
  2:'mother',
  3:'paternal_grandfather',
  4:'paternal_grandmother',
  5:'maternal_grandfather',
  6:'maternal_grandmother',
  7:'paternal_aunt_or_uncle',
  8:'maternal_aunt_or_uncle',
  9:'paternal_cousin',
  10:'maternal_cousin',
  11:'sibling_full',
  12:'sibling_half',
  13:'spouse',
  14:'offspring'
}
N_RELATIONSHIPS = len(GRAPH_NODE_STRUCTURE)


class DataFetch():

    def __init__(self, maskfile, featfile, statfile, params):
        self.params = params 
        
        feat_df = pd.read_csv(featfile)
        self.label_key              = feat_df[feat_df['type']=='label']['name'].tolist()[0]        
        self.static_features        = feat_df[feat_df['type']=='static']['name'].tolist()
        self.edge_features          = feat_df[feat_df['type']=='edge']['name'].tolist()
        #NB: some gnn layers only support a single edge weight
        del feat_df
        
        self.stat_df = pd.read_csv(statfile, index_col='target_node_id')
        # HACK: build masked target row for future
        self.zeros_array = np.zeros((1, len(self.static_features)))

        mask_df = pd.read_csv(maskfile)
        self.train_patient_list               = torch.tensor(mask_df[mask_df['train']==0]['node_id'].to_numpy())
        self.validate_patient_list            = torch.tensor(mask_df[mask_df['train']==1]['node_id'].to_numpy())
        self.test_patient_list                = torch.tensor(mask_df[mask_df['train']==2]['node_id'].to_numpy())
        del mask_df
        
        self.num_samples_train_majority_class, self.num_samples_train_minority_class = self.stat_df[self.stat_df.train==0 & self.stat_df.relationship_detail.isna()][self.label_key].value_counts().values
        self.num_samples_valid_majority_class, self.num_samples_valid_minority_class = self.stat_df[self.stat_df.train==1 & self.stat_df.relationship_detail.isna()][self.label_key].value_counts().values

        
    def construct_patient_graph(self, patient):
        
        # extract target patient info
        static_subset = self.stat_df.loc[patient,self.static_features+['relationship_detail']]
        label_subset = self.stat_df.loc[patient,[self.label_key]+['relationship_detail']]
        
        if self.params['mask_target']=='True': 
            x_static = self.zeros_array
        else:
            x_static = static_subset[static_subset.relationship_detail.isna()][self.static_features].values
        y = label_subset[label_subset.relationship_detail.isna()][self.label_key].values
        
        # build graph and calculate aggregated features for clusters
        aggregated_data_list = []
        static_subset_values = static_subset.values
        if self.params['aggr_func']=='mean':  
            aggr_func=np.mean
        elif self.params['aggr_func']=='min':  
            aggr_func=np.min
        elif self.params['aggr_func']=='max':  
            aggr_func=np.max
        
        for relationship in GRAPH_NODE_STRUCTURE.values():
            # this code works only if relationship_detail is the last column
            relationship_data = static_subset_values[static_subset_values[:, -1] == relationship][:, :-1]
            # if cluster is empty create ghost node with no info
            if relationship_data.size!=0:
                aggr_data = np.array([np.apply_along_axis(aggr_func, axis=0, arr=relationship_data)])
                aggregated_data_list.append(aggr_data)
            else:
                aggregated_data_list.append(self.zeros_array)
        
        aggregated_data = np.concatenate(aggregated_data_list, axis=0)
        x_static = np.concatenate([x_static, aggregated_data], axis=0)
        
        # construct pytorch tensors
        x_static = torch.tensor(x_static, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)

        # build edge connections
        # look at GRAPH_NODE_STRUCTURE for more info
        edges = [[1,0],[2,0],[3,1],[4,1],[5,2],[6,2],[7,1],[9,7],[8,2],[10,8],[11,0],[12,0],[13,0],[14,0]]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # create graph
        if self.params['directed']=='True': 
            data = torch_geometric.data.Data(x=x_static, y=y, edge_index=edge_index, directed=True)
        else:
            data = torch_geometric.data.Data(x=x_static, y=y, edge_index=edge_index, directed=False)
        data.target_index = 0
        
        return data

class GraphData(GraphDataset):

    def __init__(self, patient_list, fetch_data, params):
        self.patient_list = patient_list
        self.num_target_patients = len(patient_list)
        self.fetch_data = fetch_data

          
    def __getitem__(self, patients):
        # returns multiple patient graphs by constructing a pytorch geometric Batch object
        batch_patient_list = self.patient_list[patients]
        data_list = []

        for patient in batch_patient_list:
            patient_graph = self.fetch_data.construct_patient_graph(patient.item())
            data_list.append(patient_graph)

        batch_data = Batch.from_data_list(data_list)
        return batch_data
  
    def __len__(self):
        return self.num_target_patients


def get_batch_and_loader(patient_list, fetch_data, params, shuffle=False):
    
    dataset = GraphData(patient_list, fetch_data, params)

    if shuffle:
        sample_order = sampler.RandomSampler(dataset)
    else:
        sample_order = sampler.SequentialSampler(dataset)

    Sampler = sampler.BatchSampler(
        sample_order,
        batch_size=params['batchsize'],
        drop_last=False)

    loader = DataLoader(dataset, sampler=Sampler)
    return dataset, loader


