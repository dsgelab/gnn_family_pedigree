
import pandas as pd
import numpy as np
import gc

PROJECT_PATH   = "/data/projects/project_GNN/gnn_family_pedigree/"
RELATIVES_PATH = "/data/projects/project_SophieAndZhiyu/Relatives/family_relationships.wMarrigeChild"
STATFILE_PATH  = PROJECT_PATH + "data/statfile.csv"

# fetch data
df = pd.read_csv(RELATIVES_PATH)
statfile = pd.read_csv(STATFILE_PATH)
print(f'starting with {df.shape[0]} total edges')

# remove non graph patients
#to_keep = statfile[statfile.graph==1]['FINREGISTRYID'].tolist()
#df = df[ (df.ID1.isin(to_keep)) ].reset_index(drop=True)
#df = df[ (df.ID2.isin(to_keep)) ].reset_index(drop=True)
# all passed the QC step with new minimal_pheno file

# drop ambiguous relatives
df = df[df.completeness != 'unknown'].reset_index(drop=True)
print(f'remove unknown edges: {df.shape[0]} edges available now')

# add graph edge weights
WEIGHT_MAP = {
    'parent_na':0.5, 
    'sibling_full':0.5, 
    'offspring_na':0.5,
    'spouse_na':0.3,
    'grandparent_na':0.25, 
    'aunt_or_uncle_full':0.25, 
    'cousin_full':0.125, 
    'sibling_half':0.25,
    'aunt_or_uncle_half':0.125}

df['relationship_type'] = df.relationship + '_' + df.completeness 
df['weight'] = df.relationship_type.map(WEIGHT_MAP)

# remove missing IDs
df = df[ df.ID1.notna() ].reset_index(drop=True)
df = df[ df.ID2.notna() ].reset_index(drop=True)

# map to node ids
NODE_MAP = dict(zip(statfile['FINREGISTRYID'], statfile['node_id']))
df['node1'] = df['ID1'].map(NODE_MAP).astype('int64')
df['node2'] = df['ID2'].map(NODE_MAP).astype('int64')

# add target patient info
target_patients = statfile[statfile.target==1]['FINREGISTRYID'].tolist()
df['target_patient'] = np.where(df.node1.isin(target_patients), df.node1, -1)
# find smallest and biggest target subgraphs:
print(df.groupby('target_patient').value_counts())

# save results
df_edge = df[['node1','node2','target_patient','relationship','relationship_type','weight']]
df_edge.to_csv(PROJECT_PATH+"data/edgefile.csv", index=None)

gc.collect()
