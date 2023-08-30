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
    """Class for fetching and formatting data

    Expects a tensor list of patients encoded using the numerical node_ids

    Assumes maskfile, statfile rows are indexed in order of these node_ids (0, 1, ... num_samples)
    and they include data for both the target and graph samples (retrieve data using .iloc)
    
    The edgefile only needs to include data for the target samples, and is indexed
    using the node_ids (retrieve data using .loc)

    Note: the featfile has exactly one label, corresponding to the label column name in the statfile
    Note: if the input is a directed graph the code converts it to an undirected graph

    Args:
        maskfile, featfile, statfile and edgefile are filepaths to csv files
        sqlpath is the path to the sql database
    """
    def __init__(self, maskfile, featfile, statfile, edgefile, params):

        feat_df = pd.read_csv(featfile)
        self.params = params 
        self.label_key              = feat_df[feat_df['type']=='label']['name'].tolist()[0]        
        self.static_features        = feat_df[feat_df['type']=='static']['name'].tolist()
        self.edge_features          = feat_df[feat_df['type']=='edge']['name'].tolist()
        # some gnn layers only support a single edge weight
        if params['gnn_layer'] in ['gcn', 'graphconv']: self.edge_features=['weight']

        stat_df = pd.read_csv(statfile)
        self.label_data     = torch.tensor(stat_df[self.label_key].to_numpy(), dtype=torch.float)
        self.static_data    = torch.tensor(stat_df[self.static_features].values, dtype=torch.float)
       
        mask_df = pd.read_csv(maskfile)
        self.train_patient_list               = torch.tensor(mask_df[mask_df['train']==0]['node_id'].to_numpy())
        self.validate_patient_list            = torch.tensor(mask_df[mask_df['train']==1]['node_id'].to_numpy())
        self.test_patient_list                = torch.tensor(mask_df[mask_df['train']==2]['node_id'].to_numpy())

        self.edge_df = pd.read_csv(edgefile)
        # crete two datasets: 
        # 1. for connecting nodes and avoid double edges 
        edges_to_keep = self.edge_df.relationship_type.isin(['parent_na','offspring_na','sibling_full','sibling_na'])
        self.edge_connections = self.edge_df[edges_to_keep]
        # 2. for extracting the node features
        self.edge_df = self.edge_df.groupby('target_patient').agg(list)
    
    def get_static_data(self, patients):
        x_static = self.static_data[patients]
        y = self.label_data[patients]
        return x_static, y

    def get_relatives(self, patients):
        nodes_included = torch.tensor(list(set([i for list in self.edge_df.loc[patients]['node1'].to_list() for i in list] + [i for list in self.edge_df.loc[patients]['node2'].to_list() for i in list])))      
        return nodes_included

    def construct_patient_graph(self, patient, all_relatives, all_x_static, all_y):
        """Creates a re-indexed pytorch geometric data object for the patient
        """
        # order nodes and get indices in all_relatives to retrieve feature data
        node_ordering = np.asarray(list(set(self.edge_df.loc[patient].node1 + self.edge_df.loc[patient].node2)))
        node_indices = [list(all_relatives.tolist()).index(value) for value in node_ordering]
        NODE_MAP = dict(zip(node_ordering, node_indices))
        x_static = all_x_static[node_indices]
        # mask target patient with vector of all -1
        target_index = torch.tensor(list(node_ordering.tolist()).index(patient))
        if self.params['mask_target'] == 'True':
            x_static[target_index] = torch.full( (1,len(self.static_features)),-1)
        y = all_y[list(all_relatives.tolist()).index(patient)] 
        
        # prepare edge information
        edges_to_use = self.edge_connections[ (self.edge_connections.node1.isin(NODE_MAP.keys())) & (self.edge_connections.node2.isin(NODE_MAP.keys()))]
        node1 = edges_to_use['node1'].map(NODE_MAP).tolist()
        node2 = edges_to_use['node2'].map(NODE_MAP).tolist()
        unique_edges = list(set(tuple(sorted(tpl)) for tpl in [(node1,node2)]))
        edge_index = torch.tensor(unique_edges, dtype=torch.long)
        # edge_weight = torch.t(torch.tensor(self.edge_df.loc[patient][self.edge_features], dtype=torch.float))

        # create graph
        if self.params['use_edge'] == 'True':
            data = torch_geometric.data.Data(x=x_static, y=y, edge_index=edge_index, edge_attr=edge_weight)
        if self.params['use_edge'] == 'False':
            data = torch_geometric.data.Data(x=x_static, y=y, edge_index=edge_index)
        transform = torch_geometric.transforms.ToUndirected()
        data = transform(data)
        data.target_index = target_index

        return data


class GraphData(GraphDataset):
    def __init__(self, patient_list, fetch_data):
        """Loads a batch of multiple patient graphs
        """
        self.patient_list = patient_list
        self.num_target_patients = len(patient_list)
        self.fetch_data = fetch_data

    def __getitem__(self, patients):
        # returns multiple patient graphs by constructing a pytorch geometric Batch object
        batch_patient_list = self.patient_list[patients]
        data_list = []

        # it's more efficient to fetch feature data for all patients and their relatives, and then split into separate graphs
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

