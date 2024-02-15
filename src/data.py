import torch
from torch.utils.data import Dataset, sampler

import torch_geometric
from torch_geometric.data import Dataset as GraphDataset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

import pandas as pd
import numpy as np
import time
import multiprocessing

GRAPH_NODE_STRUCTURE = {
    'target': 0,
    'father': 1,
    'mother': 2,
    'paternal_grandfather': 3,
    'paternal_grandmother': 4,
    'maternal_grandfather': 5,
    'maternal_grandmother': 6,
    'paternal_aunt_or_uncle': 7,
    'maternal_aunt_or_uncle': 8,
    'paternal_cousin': 9,
    'maternal_cousin': 10,
    'sibling_full': 11,
    'sibling_half': 12,
    'spouse': 13,
    'offspring': 14
}
N_RELATIONSHIPS = len(GRAPH_NODE_STRUCTURE)
RELATIONSHIPS_SET = set(GRAPH_NODE_STRUCTURE.values())

class DataFetch():

    def __init__(self, maskfile, featfile, statfile, params):
        self.params = params 
        
        feat_df = pd.read_csv(featfile)
        self.label_key              = feat_df[feat_df['type']=='label']['name'].tolist()[0]        
        self.static_features        = feat_df[feat_df['type']=='static']['name'].tolist()
        self.edge_features          = feat_df[feat_df['type']=='edge']['name'].tolist()
        #NB: some gnn layers only support a single edge weight
        del feat_df
        
        self.stat_df = pd.read_csv(statfile)
        self.stat_df = self.stat_df[self.stat_df.relationship_detail!='sibling_unknown']
        self.stat_df.loc[self.stat_df.relationship_detail.isna(),'relationship_detail'] = 'target'       
        self.stat_df.relationship_detail = self.stat_df.relationship_detail.map(GRAPH_NODE_STRUCTURE).astype(int)
        print('loaded statfile, aggregating relationship cliusters')
        t = time.time()
        if params['aggr_func']=='max':
            self.stat_df = self.stat_df.groupby(['target_node_id', 'relationship_detail']).max().reset_index()
        elif params['aggr_func']=='min':
            self.stat_df = self.stat_df.groupby(['target_node_id', 'relationship_detail']).min().reset_index()
        elif params['aggr_func']=='sum':
            self.stat_df = self.stat_df.groupby(['target_node_id', 'relationship_detail']).sum().reset_index()
        elif params['aggr_func']=='mean':
            self.stat_df = self.stat_df.groupby(['target_node_id', 'relationship_detail']).mean().reset_index()
        self.stat_df = self.stat_df.set_index('target_node_id')
        print('completed in '+str(time.time()-t)+'seconds')
        # prepare extra info
        self.masked_target = np.zeros(len(self.stat_df.columns))
        self.N_COLS = self.stat_df.shape[1]
        self.col_idx = self.stat_df.columns.get_loc('relationship_detail')

        mask_df = pd.read_csv(maskfile)
        self.train_patient_list               = torch.tensor(mask_df[mask_df['train']==0]['node_id'].to_numpy())
        self.validate_patient_list            = torch.tensor(mask_df[mask_df['train']==1]['node_id'].to_numpy())
        self.test_patient_list                = torch.tensor(mask_df[mask_df['train']==2]['node_id'].to_numpy())
        del mask_df
        
        self.num_samples_train_majority_class, self.num_samples_train_minority_class = self.stat_df[self.stat_df.train==0 & self.stat_df.relationship_detail.isna()][self.label_key].value_counts().values
        self.num_samples_valid_majority_class, self.num_samples_valid_minority_class = self.stat_df[self.stat_df.train==1 & self.stat_df.relationship_detail.isna()][self.label_key].value_counts().values


class GraphData(GraphDataset):

    def __init__(self, patient_list, fetch_data, params):
        self.patient_list = patient_list
        self.num_target_patients = len(patient_list)
        self.fetch_data = fetch_data
        if params['mask_target']=='True':
            self.mask_target=True
        else:
            self.mask_target=False
        if params['directed']=='True':
            self.directed=True
        else:
            self.directed=False
        self.static_features = self.fetch_data.static_features
        self.label_key = self.fetch_data.label_key
    
    def construct_patient_graph(self, patient):
        patient_data = self.fetch_data.stat_df.loc[patient]
        if self.mask_target==True: 
            patient_data[patient_data.relationship_detail==0] == self.fetch_data.masked_target
        # create ghost nodes if relationship cluster is missing, then sort
        new_rows=[]
        if patient_data.shape[0]!=N_RELATIONSHIPS:
            current_set = set(patient_data.relationship_detail.tolist())
            new_rows = np.zeros( (N_RELATIONSHIPS-len(current_set), self.fetch_data.N_COLS) ) 
            new_rows[:,self.fetch_data.col_idx] = np.array(list(RELATIONSHIPS_SET - current_set))
            patient_data = pd.concat([patient_data,pd.DataFrame(new_rows, columns=patient_data.columns)], ignore_index=True)
        patient_data = patient_data.sort_values(by='relationship_detail').reset_index(drop=True)
        # construct pytorch tensors
        x_static = torch.tensor(patient_data[self.static_features].values, dtype=torch.float)
        y = torch.tensor(patient_data.loc[patient_data.relationship_detail==0,self.label_key].values, dtype=torch.float)
        # build edge connections, look at GRAPH_NODE_STRUCTURE for more info
        edges = [[1,0],[2,0],[3,1],[4,1],[5,2],[6,2],[7,1],[9,7],[8,2],[10,8],[11,0],[12,0],[13,0],[14,0]]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()        
        # create torch geometric graph object
        if self.directed==True: 
            data = torch_geometric.data.Data(x=x_static, y=y, edge_index=edge_index, directed=True)
        else:
            data = torch_geometric.data.Data(x=x_static, y=y, edge_index=edge_index, directed=False)
        data.target_index = 0       
        return data
    
    def __getitem__(self, patients):
        batch_patient_list = self.patient_list[patients]
        data_list = []       

        for patient in batch_patient_list:
            patient_graph = self.construct_patient_graph(patient.item())
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


