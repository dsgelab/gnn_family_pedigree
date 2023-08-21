
import pandas as pd
import gc

PROJECT_PATH   = "/data/projects/project_GNN/gnn_family_pedigree/"
RELATIVES_PATH = "/data/projects/project_SophieAndZhiyu/Relatives/family_relationships.csv"
STATFILE_PATH  = PROJECT_PATH + "data/statfile.csv"

# fetch data
df = pd.read_csv(RELATIVES_PATH)
statfile = pd.read_csv(STATFILE_PATH)
graph = statfile[statfile['graph']==1]['FINREGISTRYID'].tolist()

# drop ambiguous relatives
df = df[df.completeness != 'unknown'].reset_index(drop=True)

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

# map to node ids
NODE_MAP = dict(zip(statfile['FINREGISTRYID'], statfile['node_id']))
df['node1'] = df['ID1'].map(NODE_MAP)
df['node2'] = df['ID2'].map(NODE_MAP)

# remove missing IDs
df = df.loc[df['node1'].notna()].reset_index(drop=True)
df = df.loc[df['node2'].notna()].reset_index(drop=True)

# save results
df_edge = df[['node1','node2','weight','relationship_type']]
df_edge.to_csv(PROJECT_PATH+"data/edgefile.csv", index=None)

gc.collect()
