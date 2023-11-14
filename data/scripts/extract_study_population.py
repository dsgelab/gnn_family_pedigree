
# PREPARE ENVIRONMENT

# run this in ePouta: 
# source /data/projects/project_GNN/envs/graphml/bin/activate

import pandas as pd
import numpy as np
import gc
from itertools import compress
from sklearn.model_selection import train_test_split

# starting from minimal_phenotype in Finregistry extract the study population
# then adding extra information from Zhiyu file
EXTRA_FEATURES = "/data/projects/project_GNN/AllIndFeat"
PROJECT_PATH = "/data/projects/project_GNN/gnn_family_pedigree/"
MIN_PHENO_PATH = "/data/processed_data/minimal_phenotype/minimal_phenotype_2023-08-14.csv"
RELATIVES_PATH = "/data/projects/project_SophieAndZhiyu/Relatives/family_relationships.wMarrigeChild"

# FETCH DATA
df = pd.read_csv(MIN_PHENO_PATH, sep = ',', usecols=["FINREGISTRYID","INDEX_PERSON","DATE_OF_BIRTH","SEX","MOTHER_TONGUE","EMIGRATED"])
df = df.rename(str.lower,axis='columns').rename(columns={'finregistryid':'FINREGISTRYID'})
N_START = df.shape[0]

# EXCLUSION CRITERIA: 
#	missing age or sex
# born before 1920

print(f'starting from {N_START} patients available in Finregistry')

df['birth_year'] = df.date_of_birth.str[:4].astype('int64')
to_exclude = ( (df.date_of_birth.isna()) | (df.sex.isna()) | (df.birth_year<1920) )
df = df[~to_exclude]
N_INCLUDED = N_START - sum(to_exclude)
print(f'after exclusion criteria {N_INCLUDED} patients are going to be used in the study')

# assign node_id to those patients
df['node_id'] = np.arange(N_INCLUDED).astype('int64')

# TARGET PATIENT DEFINITION:
# -	 index person
# -  mother-tongue finnish or svedish
# -  has both parents 
# -  didn't emigrate
# -  is born between 1970 and 1990
# -  has both parents
# -  has at least 1 grandparent

relatives = pd.read_csv(RELATIVES_PATH, sep = ',')
# remove non eligible people
eligible = df.FINREGISTRYID.tolist()
relatives = relatives[ (relatives.ID1.isin(eligible)) & (relatives.ID2.isin(eligible)) ] 

# check parental connections
print('checking parental connections')
has_mother = set(relatives[(relatives.relationship == 'parent') & (relatives.relationship_detail == 'mother')].ID1.tolist())
has_father = set(relatives[(relatives.relationship == 'parent') & (relatives.relationship_detail == 'father')].ID1.tolist())
has_grandparent = set(relatives[relatives.relationship == 'grandparent'].ID1.tolist())
has_required_connections = list(set.intersection(has_mother, has_father, has_grandparent))
print(f'{len(has_required_connections)} have 2 parents and at least 1 grandparent')

df["target"] = 0
is_target = ( 
  (df.index_person==1) & 
  (df.mother_tongue.isin(['fi','sv'])) & 
  ((df.emigrated == 0) | (df.emigrated.isna())) &
  ((df.birth_year>=1970) & (df.birth_year<=1990)) &
  (df.FINREGISTRYID.isin(has_required_connections))
)
  
df.loc[is_target,"target"] = 1
print(f'{sum(is_target)} patients are going to be used as target patients')

# TRAIN/TEST SPLIT 
# 70% train, 10% valid, 20% test 

# if not target put -1 
df["train"] = -1

train_valid_test_split = [0.7,0.1,0.2]
df.loc[df.target==1,"train"] = np.random.choice([0,1,2], sum(is_target), p=train_valid_test_split)
df = df.sort_values(by=['node_id'])

print(f'there are {sum(df.train==-1)} non target patients')
print(f'there are {sum(df.train==0)} training target patients')
print(f'there are {sum(df.train==1)} validation target patients')
print(f'there are {sum(df.train==2)} test target patients')

# create maskfile
MASKFILE_COLS = ['FINREGISTRYID','node_id','target','train']
df[MASKFILE_COLS].to_csv(PROJECT_PATH+"data/maskfile.csv", index=False)

# add extra features and final label then save to statfile
print('adding extra features')
# extract 5 diseases from Sophie's paper 
DISEASE_LIST = ['I9_CHD_nEvent','T2D_nEvent','J10_ASTHMA_nEvent','F5_DEPRESSIO_nEvent','C3_COLORECTAL_nEvent']
extra = pd.read_csv(EXTRA_FEATURES, sep=',',usecols=['FINREGISTRYID']+DISEASE_LIST) 
df = df.merge(extra, how='left',on='FINREGISTRYID')
df['CHD_binary']        = (df['I9_CHD_nEvent']>0).astype('int64')
df['T2D_binary']        = (df['T2D_nEvent']>0).astype('int64')
df['ASTHMA_binary']     = (df['J10_ASTHMA_nEvent']>0).astype('int64')
df['DEPRESSION_binary'] = (df['F5_DEPRESSIO_nEvent']>0).astype('int64')
df['COL_CANC_binary']   = (df['C3_COLORECTAL_nEvent']>0).astype('int64')

# save results
df.to_csv(PROJECT_PATH+"data/statfile.csv",index=False)

gc.collect()
print('finished')
