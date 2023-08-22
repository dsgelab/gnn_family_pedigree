import torch
from torch.utils.data import Dataset, sampler

import torch_geometric
from torch_geometric.data import Dataset as GraphDataset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

import pandas as pd
import numpy as np
from random import choices

# import networkx as nx
# import matplotlib.pyplot as plt


class DataFetch():
    """Class for fetching patient data

    Expects a list of patients encoded using the node_ids

    The edgefile only needs to include data for the target samples, and is indexed
    using the node_ids (retrieve data using .loc)

    Note: the featfile has exactly one label, corresponding to the label column name in the statfile
    Note: if the input is a directed graph the code converts it to an undirected graph

    Args:
        maskfile, featfile, statfile and edgefile (str): filepaths to respective csv files
        params (list): contains all the parameter specification used in the project
        patient_list (torch.tensor):

    Methods:
        get_all_relatives(): takes tensor of target patients node_ids in input and gives a tensor of relatives node_ids as output
        get_all_static_data(): takes tensor of patients node_ids in input and gives covariate information as output 
    """ 

    def __init__(self, maskfile, featfile, statfile, edgefile, params):

        self.params = params

        feat_df = pd.read_csv(featfile) 
        self.label_key              = feat_df[feat_df['type']=='label']['name'].tolist()[0]        
        self.static_features        = feat_df[feat_df['type']=='static']['name'].tolist()
        self.edge_features          = feat_df[feat_df['type']=='edge']['name'].tolist()
        # gcn and graphconv only support a single edge weight
        if params['gnn_layer'] in ['gcn', 'graphconv']: self.edge_features=['weight']

        stat_df = pd.read_csv(statfile)
        self.label_data     = torch.tensor(stat_df[self.label_key].to_numpy(), dtype=torch.float)
        self.static_data    = torch.tensor(stat_df[self.static_features].to_numpy(), dtype=torch.float)

        mask_df = pd.read_csv(maskfile)
        self.train_patient_list               = torch.tensor(mask_df[mask_df['train']==0]['node_id'].to_numpy(),dtype=torch.long)
        self.validate_patient_list            = torch.tensor(mask_df[mask_df['train']==1]['node_id'].to_numpy(),dtype=torch.long)
        self.test_patient_list                = torch.tensor(mask_df[mask_df['train']==2]['node_id'].to_numpy(),dtype=torch.long)

        self.edge_df = pd.read_csv(edgefile)
        self.edge_df = self.edge_df.groupby('target_id').agg(list)
        
    # NB: is more efficient to retrive all information at once for every target patient
    
    def get_all_relatives(self, patient_list):
        nodes_included = torch.tensor(list(set(
          [i for list in self.edge_df.loc[patient_list]['node1'].to_list() for i in list] + 
          [i for list in self.edge_df.loc[patient_list]['node2'].to_list() for i in list]
          )))      
        return nodes_included
      
    def get_all_static_data(self, patient_list):
        # retrive static info for all nodes included in patient_list graphs
        x_static = self.static_data[patient_list]
        y = self.label_data[patient_list]
        return x_static, y


class GraphData(GraphDataset):
    def __init__(self, patient_list, fetch_data):
        """Loads a batch of multiple patient graphs
        
        Args: 
          patient_list (torch.Tensor): list of all target patients to consider
          fetch_data (DataFetch obj): fetched information about every patient
        """
        self.patient_list = patient_list
        self.num_target_patients = len(patient_list)
        self.fetch_data = fetch_data
        
    def construct_patient_graph(self, target_patient, all_relatives, all_x_static, all_y):
        """Creates a pytorch_geometric data object for one target patient
        """
        
        # get all unique nodes to be used in the graph, use np.asarray for performance
        nodes_ids = np.asarray(list(set(
          self.fetch_data.edge_df.loc[target_patient].node1 + 
          self.fetch_data.edge_df.loc[target_patient].node2
          )))
        node_indices = [list(all_relatives.tolist()).index(value) for value in nodes_ids]
        
        # if required perform masking (put -1) of target patient info
        x_static = all_x_static[node_indices]
        target_index = torch.tensor(list(node_ordering.tolist()).index(target_patient))
        if self.fetch_data.params['mask_target']=='True': 
          x_static[target_index] = torch.full( size = (1,len(self.static_features)), value = -1)
        y = all_y[list(all_relatives.tolist()).index(target_patient)] 

        # prepare weight info
        node1 = [list(nodes_ids.tolist()).index(value) for value in self.edge_df.loc[target_patient].node1]
        node2 = [list(nodes_ids.tolist()).index(value) for value in self.edge_df.loc[target_patient].node2]
        edge_index = torch.tensor([node1,node2], dtype=torch.long)
        if self.fetch_data.params['use_edge'] == 'True':
            edge_weight = torch.t(torch.tensor(self.edge_df.loc[target_patient][self.edge_features], dtype=torch.float))
        elif self.fetch_data.params['use_edge'] == 'False':
            edge_weight = torch.full(size = (self.edge_df.loc[target_patient].shape[0],1), value = 1)
        # create graph data
        data = torch_geometric.data.Data(x=x_static, y=y, edge_index=edge_index, edge_attr=edge_weight)
          
        transform = torch_geometric.transforms.ToUndirected(reduce='mean')
        final_data = transform(data)
        # add info on target patient node
        final_data.target_index = target_index
        
        return final_data

    def __getitem__(self, patient_to_batch):
        """Creates a pytorch geometric Batch object for multiple target patients
        """
        batch_patient_list = self.patient_list[patient_to_batch]
        data_list = []

        all_relatives           = self.fetch_data.get_all_relatives(batch_patient_list)
        all_x_static, all_y     = self.fetch_data.get_all_static_data(all_relatives)

        for patient in batch_patient_list:
            patient_graph = construct_patient_graph(patient.item(), all_relatives, all_x_static, all_y)
            data_list.append(patient_graph)

        batch_data = Batch.from_data_list(data_list)
        return batch_data
  
    def __len__(self):
        return self.num_target_patients


def get_batch_and_loader(patient_list, fetch_data, params, shuffle=True):
    """Prepare the graph dataset to be used for training

    Args:
        patient_list (torch.tensor): list of patients to load data for 
        fetch_data (DataFetch):  class defined in data.py
        params: user requests, loaded using argparser
        shuffle: samples in random order if true

    Returns:
        dataset (torch_geometric.data.Dataset)
        loader ()    
    """  
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

