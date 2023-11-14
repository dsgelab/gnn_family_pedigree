import torch
from torch.utils.data import Dataset, sampler

import torch_geometric
from torch_geometric.data import Dataset as GraphDataset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

import pandas as pd
import numpy as np
from random import choices

class DataFetch():

    def __init__(self, maskfile, featfile, statfile, edgefile, params):

        self.params = params 

        feat_df = pd.read_csv(featfile)
        self.label_key              = feat_df[feat_df['type']=='label']['name'].tolist()[0]        
        self.static_features        = feat_df[feat_df['type']=='static']['name'].tolist()
        self.edge_features          = feat_df[feat_df['type']=='edge']['name'].tolist()
        #NB: some gnn layers only support a single edge weight
        del feat_df

        stat_df = pd.read_csv(statfile)
        self.label_data     = torch.tensor(stat_df[self.label_key].to_numpy(), dtype=torch.float)
        self.static_data    = torch.tensor(stat_df[self.static_features].values, dtype=torch.float)
        del stat_df
        
        mask_df = pd.read_csv(maskfile)
        self.train_patient_list               = torch.tensor(mask_df[mask_df['train']==0]['node_id'].to_numpy())
        self.validate_patient_list            = torch.tensor(mask_df[mask_df['train']==1]['node_id'].to_numpy())
        self.test_patient_list                = torch.tensor(mask_df[mask_df['train']==2]['node_id'].to_numpy())
        self.num_samples_train_minority_class = torch.sum(self.label_data[self.train_patient_list]==1).item()
        self.num_samples_train_majority_class = torch.sum(self.label_data[self.train_patient_list]==0).item()
        self.num_samples_valid_minority_class = torch.sum(self.label_data[self.validate_patient_list]==1).item()
        self.num_samples_valid_majority_class = torch.sum(self.label_data[self.validate_patient_list]==0).item()
        del mask_df
        
        self.edge_df = pd.read_csv(edgefile)
        self.edge_df = self.edge_df.groupby('target_patient').agg(list)
        
    def get_static_data(self, patients):
        x_static = self.static_data[patients]
        y = self.label_data[patients]
        return x_static, y

    def get_relatives(self, patients):
        nodes_included = torch.tensor(list(set([i for list in self.edge_df.loc[patients]['node1'].to_list() for i in list] + [i for list in self.edge_df.loc[patients]['node2'].to_list() for i in list])))      
        return nodes_included

    def construct_patient_graph(self, patient, all_relatives, all_x_static, all_y):
      
        # get nodes and get indices in all_relatives to retrieve feature data
        node_ordering = np.asarray(list(set(self.edge_df.loc[patient].node1 + self.edge_df.loc[patient].node2)))
        node_indices = [list(all_relatives.tolist()).index(value) for value in node_ordering]
        x_static = all_x_static[node_indices]
        y = all_y[list(all_relatives.tolist()).index(patient)] 

        # mask target patient with vector of all -1
        target_index = node_ordering.tolist().index(patient)
        if self.params['mask_target']=='True': 
            x_static[target_index] = torch.full( (1,len(self.static_features)),-1)

        # extract edge informations
        node1 = [list(node_ordering.tolist()).index(value) for value in self.edge_df.loc[patient].node1]
        node2 = [list(node_ordering.tolist()).index(value) for value in self.edge_df.loc[patient].node2]
        edge_index = torch.tensor([node1,node2], dtype=torch.long)
        edge_weight = torch.t(torch.tensor(self.edge_df.loc[patient][self.edge_features], dtype=torch.float))
        
        # create graph
        if self.params['directed']=='True': 
            data = torch_geometric.data.Data(x=x_static, y=y, edge_index=edge_index, edge_attr=edge_weight, directed=True)
        else:
            data = torch_geometric.data.Data(x=x_static, y=y, edge_index=edge_index, edge_attr=edge_weight, directed=False)
        data.target_index = torch.tensor(target_index)
        
        return data

class GraphData(GraphDataset):

    def __init__(self, patient_list, fetch_data):
        self.patient_list = patient_list
        self.num_target_patients = len(patient_list)
        self.fetch_data = fetch_data

    def __getitem__(self, patients):
        # returns multiple patient graphs by constructing a pytorch geometric Batch object
        batch_patient_list = self.patient_list[patients]
        data_list = []

        #NB: it's more efficient to fetch feature data for all patients and their relatives, and then split into separate graphs
        all_relatives           = self.fetch_data.get_relatives(batch_patient_list)
        all_x_static, all_y     = self.fetch_data.get_static_data(all_relatives)

        for patient in batch_patient_list:
            patient_graph = self.fetch_data.construct_patient_graph(patient.item(), all_relatives, all_x_static, all_y)
            data_list.append(patient_graph)

        batch_data = Batch.from_data_list(data_list)
        return batch_data
  
    def __len__(self):
        return self.num_target_patients


def get_batch_and_loader(patient_list, fetch_data, params, shuffle=True):
    
    dataset = GraphData(patient_list, fetch_data)

    if shuffle:
        sample_order = sampler.RandomSampler(dataset)
    else:
        sample_order = sampler.SequentialSampler(dataset)

    Sampler = sampler.BatchSampler(
        sample_order,
        batch_size=params['batchsize'],
        drop_last=False)

    loader = DataLoader(dataset, sampler=Sampler, num_workers=params['num_workers'])
    return dataset, loader

