
import pandas as pd
import numpy as np
import gc

PROJECT_PATH   = "/data/projects/project_GNN/gnn_family_pedigree/"
RELATIVES_PATH = "/data/projects/project_GNN/TarEdgeFile"
MASKFILE_PATH  = PROJECT_PATH + "data/maskfile.csv"

# FETCH DATA
maskfile = pd.read_csv(MASKFILE_PATH)
df = pd.read_csv(RELATIVES_PATH, sep='\t')

# remove missing values
df = df.dropna()

# map to node ids
print('start node mapping')
NODE_MAP = dict(zip(maskfile['FINREGISTRYID'], maskfile['node_id']))

# NB: switching the nodes to keep the direction of the edges the "genetic way"
df['node2'] = df['ID1'].map(NODE_MAP)
df['node1'] = df['ID2'].map(NODE_MAP)
df['target_patient'] = df.Target.map(NODE_MAP)

# remove non graph patients
# NB: they will be missing value
PRE = df.shape[0]
df = df.dropna(subset=['node1','node2','target_patient'])
POST = df.shape[0]
print(f'removing {POST-PRE} non graph patient rows')

# QUALITY CHECK:
target_left = maskfile[maskfile.target==1].node_id.tolist()
target_right = df.target_patient.tolist()
check = len(set(target_left)-set(target_right))
print(f'the following number should be 0: {check}, if not there is someone missing in edgefile')

# keep only GNN target patients (see statfile definition)
to_keep = maskfile[maskfile.target==1].node_id.tolist()
df = df[df.target_patient.isin(to_keep)]
df.sort_values(by='target_patient',inplace=True)

# add extra info
df['weight'] = 0.5
df.rename(columns={'relationship_detail':'relationship'},inplace=True)

# save results
print("saving results")
df['node1'] = df.node1.astype('int64')
df['node2'] = df.node2.astype('int64')
df['target_patient'] = df.target_patient.astype('int64')
df_edge = df[['node1','node2','target_patient','relationship','weight']]
df_edge.to_csv(PROJECT_PATH+"data/edgefile_onlyparents.csv", index=None)

gc.collect()
