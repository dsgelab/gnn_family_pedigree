
# PREPARE ENVIRONMENT

# run this in ePouta: 
# source /data/projects/project_GNN/envs/graphml/bin/activate

import pandas as pd
import numpy as np
import gc

# starting from minimal_phenotype in Finregistry extract the study population

MIN_PHENO_PATH = "/data/processed_data/minimal_phenotype/archive/minimal_phenotype_2022-03-28.csv"
ENDPOINTS = ...
PROJECT_PATH = "/data/projects/project_GNN/chd_prediction/"

TIME_POINT = "2010-01-01"
DAYS_TO_YEARS = 365.25

# FETCH DATA
df = pd.read_csv(MIN_PHENO_PATH, sep = ',')
N_START = df.shape[0]

# EXCLUSION CRITERIA: 
# 	non index person
# 	missing age or sex

df["graph"] = 1
print(f'starting from {N_START} patients available in Finregistry')

to_exclude = (df.date_of_birth.isna()) | (df.sex.isna()) | (df.index_person!=1)
df.loc[to_exclude, "graph"] = 0
N_INCLUDED = N_START + sum(to_exclude)
print(f'after exclusion criteria {N_INCLUDED} patients are going to be used in the study')

# assign node_id to those patients
df.loc[df.graph==0,'node_id'] = np.NaN
df.loc[df.graph==1,'node_id'] = np.arange(N_INCLUDED) + 1

# format date variables
df["birth_date"] = pd.to_datetime( df.date_of_birth,  format="%Y-%m-%d",errors="coerce" )
df["death_date"] = pd.to_datetime( df.death_date,  format="%Y-%m-%d",errors="coerce" )
time_point = pd.to_datetime( TIME_POINT,  format="%Y-%m-%d",errors="coerce" )

# extract covariates of interest
df["age"]	= round( (time_point - df.birth_date).dt.days/DAYS_TO_YEARS, 2)
df["alive"]	= np.where(df.death_date>time_point,0,1)

# TARGET PATIENT DEFINITION:
# 	experienced chd diagnosis after 2010
# 	speaks finnish or svedish

TARGET = 

print(f'{sum(TARGET)} patients are going to be used as target patient')

# save results
STATFILE_COLS = ['FINREGISTRYID','age','sex','alive','mother_tongue','emigrated','index_person','graph','node_id']
MASKFILE_COLS = ['FINREGISTRYID','graph','node_id','target','train']

df[STATFILE_COLS].to_csv(PROJECT_PATH+"data/statfile.csv", index=False)
df[MASKFILE_COLS].to_csv(PROJECT_PATH+"data/maskfile.csv", index=False)
gc.collect()
