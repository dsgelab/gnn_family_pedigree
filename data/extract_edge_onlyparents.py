
import pandas as pd
import numpy as np
import gc

PROJECT_PATH   = "/data/projects/project_GNN/gnn_family_pedigree/"
RELATIVES_PATH = "/data/projects/project_GNN/TarEdgeFile"
STATFILE_PATH  = PROJECT_PATH + "data/statfile.csv"

# FETCH DATA
statfile = pd.read_csv(STATFILE_PATH)
df = pd.read_csv(RELATIVES_PATH, sep='\t',lineterminator='\n')

# remove missing values
df = df.dropna()

# map to node ids
print('start node mapping')
NODE_MAP = dict(zip(statfile['FINREGISTRYID'], statfile['node_id']))
df['node1'] = df['ID1'].map(NODE_MAP).astype('int64')
df['node2'] = df['ID2'].map(NODE_MAP).astype('int64')
df['target_patient'] = df.Target.map(NODE_MAP).astype('int64')

# add extra info
df['weight'] = 0.5
df.rename(columns={'relationship_detail':'relationship'},inplace=True)

# save results
print("saving results")
df_edge = df[['node1','node2','target_patient','relationship','weight']]
df_edge.to_csv(PROJECT_PATH+"data/edgefile_onlyparents.csv", index=None)

gc.collect()
