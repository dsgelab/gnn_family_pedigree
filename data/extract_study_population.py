
# PREPARE ENVIRONMENT

# run this in ePouta: 
# source /data/projects/project_GNN/envs/graphml/bin/activate

import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split

# starting from minimal_phenotype in Finregistry extract the study population

MIN_PHENO_PATH = "/data/processed_data/minimal_phenotype/archive/minimal_phenotype_2022-03-28.csv"
PROJECT_PATH = "/data/projects/project_GNN/gnn_family_pedigree/"

TIME_POINT = "2010-01-01"
DAYS_TO_YEARS = 365.25

# FETCH DATA
df = pd.read_csv(MIN_PHENO_PATH, sep = ',')
N_START = df.shape[0]

# EXCLUSION CRITERIA: 
#	missing age or sex

df["graph"] = 1
print(f'starting from {N_START} patients available in Finregistry')

to_exclude = ( (df.date_of_birth.isna()) | (df.sex.isna()) )
df.loc[to_exclude, "graph"] = 0
N_INCLUDED = N_START - sum(to_exclude)
print(f'after exclusion criteria {N_INCLUDED} patients are going to be used in the study')

# assign node_id to those patients
df.loc[df.graph==0,'node_id'] = -1
df.loc[df.graph==1,'node_id'] = np.arange(N_INCLUDED) + 1
df['node_id'] = df.node_id.astype('int64')

# format date variables
df["birth_date"] = pd.to_datetime( df.date_of_birth,  format="%Y-%m-%d",errors="coerce" )
df["death_date"] = pd.to_datetime( df.death_date,  format="%Y-%m-%d",errors="coerce" )
time_point = pd.to_datetime( TIME_POINT,  format="%Y-%m-%d",errors="coerce" )

# extract covariates of interest
df["age"]	= round( (time_point - df.birth_date).dt.days/DAYS_TO_YEARS, 2)
df["alive"]	= np.where(df.death_date>time_point,0,1)

# TARGET PATIENT DEFINITION:
# 	index person
# 	speaks finnish or svedish
#   has both parents 

with open("/data/projects/project_GNN/gnn_family_pedigree/data/both_parents_list.txt") as file:
    has_both_parents = [line.strip() for line in file]

df["target"] = 0
is_target = ( (df.index_person==1) & (df.mother_tongue.isin(['fi','sw'])) & (df.FINREGISTRYID.isin(has_both_parents)) )
df.loc[is_target,"target"] = 1
print(f'{sum(is_target)} patients are going to be used as target patients')

# TRAIN/TEST SPLIT 
# 70% train, 10% valid, 20% test 

# if not target put -1 
df["train"] = -1

train_valid_test_split = [0.7,0.1,0.2]
df.loc[df.target==1,"train"] = np.random.choice([0,1,2], sum(is_target), p=train_valid_test_split)

print(f'there are {sum(df.train==0)} training target patients')
print(f'there are {sum(df.train==1)} validation target patients')
print(f'there are {sum(df.train==2)} test target patients')

# uncomment and run the following lines to extract a sample dataset
# df = df.iloc[:100_000,:]

# save results
STATFILE_COLS = ['FINREGISTRYID','age','sex','alive','mother_tongue','emigrated','index_person','graph','node_id','target']
MASKFILE_COLS = ['FINREGISTRYID','graph','node_id','target','train']

df[STATFILE_COLS].to_csv(PROJECT_PATH+"data/statfile.csv", index=False)
df[MASKFILE_COLS].to_csv(PROJECT_PATH+"data/maskfile.csv", index=False)
gc.collect()
print('finished')
