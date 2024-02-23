
# to run on ePouta:
# conda activate /data/tools/conda
# python <filename.py>

#-------------------------------------------------------------------------------
# PREPARE ENVIRONMENT
import pandas as pd
import numpy as np
import gc
from datetime import datetime

START = datetime.now()
RELATIVES_PATH = "/data/projects/project_SophieAndZhiyu/Relatives/family_relationships.wMarrigeChild"
STATFILE_PATH = "/data/projects/project_GNN/GAT_family_pedigree/data/statfile.csv"
OUTPUT_PATH = "/data/projects/project_GNN/GAT_family_pedigree/data/statfile_with_relationships.csv"

# FETCH DATA
df = pd.read_csv(STATFILE_PATH, sep = ',')
relatives = pd.read_csv(RELATIVES_PATH, sep = ',')

#-------------------------------------------------------------------------------
# DEFINE FUNCTIONS

def fix_relationship_detail(df):
    assert set(['relationship','relationship_detail']).issubset(df.columns), 'missing required columns' 
    df.loc[df.relationship=='offspring','relationship_detail'] = 'offspring'
    df.loc[df.relationship=='spouse','relationship_detail'] = 'spouse'
    df.loc[df.relationship=='sibling','relationship_detail'] = 'sibling_'+df.completeness
    df.loc[df.relationship=='cousin','relationship_detail'] = df.relationship_detail+'_cousin'
    df.loc[df.relationship=='aunt_or_uncle','relationship_detail'] = df.relationship_detail+'_aunt_or_uncle'
    mapping = {'father_aunt_or_uncle': 'paternal_aunt_or_uncle', 'mother_aunt_or_uncle': 'maternal_aunt_or_uncle'}
    df.relationship_detail = df.relationship_detail.replace(mapping)
    return df


#-------------------------------------------------------------------------------
# append info about relatives
# NB: only for targets to save space

node_mapping = {key: value for key, value in zip(df['FINREGISTRYID'], df['node_id'])}
target_list = df[df.target==1].FINREGISTRYID.tolist()

relatives = fix_relationship_detail(relatives)
relatives = relatives[relatives.ID1.isin(target_list)]
relatives = relatives[['ID1','ID2','relationship_detail']]

extra = df.merge(relatives,left_on='FINREGISTRYID',right_on='ID2',how='inner').drop(['ID2'], axis=1).rename(columns={'ID1':'target_id'})
df['target_id'] = df.FINREGISTRYID
df['relationship_detail'] = ''
df = pd.concat([df, extra])
        
# map to node ids and sort
df['target_node_id'] = df.target_id.map(node_mapping)
df = df.sort_values(by=['node_id'])

# fix relationship detail for model
df = df[df.relationship_detail!='sibling_unknown']
df.loc[df.relationship_detail=='','relationship_detail'] = 'target'     

# standardize continuous variables
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
columns_to_scale = ['birth_year','I9_CHD_nEvent']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# drop duplicates
df = df.drop_duplicates()

# save results
df.to_csv(OUTPUT_PATH,index=False)

print(f'finished in {datetime.now()-START} hr:min:sec')
gc.collect()

