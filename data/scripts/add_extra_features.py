# PREPARE ENVIRONMENT

# run this in ePouta: 
# source /data/projects/project_GNN/envs/graphml/bin/activate

import pandas as pd
import gc
import os
import time

# prepare files
print('FETCHING DATA')
statfile = pd.read_csv(
    filepath_or_buffer = '/data/projects/project_GNN/gnn_family_pedigree/data/statfile_nokids.csv', 
    usecols = ['FINREGISTRYID','node_id','birth_year','sex','I9_CHD_nEvent','CHD_binary'])
statfile = statfile[['FINREGISTRYID','node_id','birth_year','sex','I9_CHD_nEvent','CHD_binary']]

# add endpoint label
# LABEL_COLS = ['CHD_binary','T2D_binary','ASTHMA_binary','DEPRESSION_binary','COL_CANC_binary'] 

# download extra feature names 
extra_SES = pd.read_csv("/data/projects/project_GNN/Indep0.2/SES",header=None)[0].tolist()
extra_Drug = pd.read_csv("/data/projects/project_GNN/Indep0.2/Drug",header=None)[0].tolist()
extra_EndPt = pd.read_csv("/data/projects/project_GNN/Indep0.2/EndPt",header=None)[0].tolist()

# update names
extra_Drug = [el+'_OnsetAge' for el in extra_Drug] 
extra_EndPt = [el+'_OnsetAge' for el in extra_EndPt] 
extra_all = extra_SES + extra_Drug + extra_EndPt 


#-----------------------------
print('prepearing estended statfile headers')

# all 3
OUTPUT_FILE = '/data/projects/project_GNN/gnn_family_pedigree/data/nokids_extended/statfile_chd_all.csv'
output_columns = ['FINREGISTRYID','node_id','birth_year','sex','I9_CHD_nEvent','CHD_binary'] + extra_all
output_columns[output_columns.index('247_psychiatric_residential_care')] = 'psychiatric_residential_care'
output_columns[output_columns.index('247_residential_care_housing_under_65yo')] = 'residential_care_housing_under_65yo'
output = pd.DataFrame(columns=output_columns)
output.to_csv(OUTPUT_FILE, mode='w', index=False)


#-----------------------------
print('appending to extendended ALL')

OUTPUT_FILE = '/data/projects/project_GNN/gnn_family_pedigree/data/nokids_extended/statfile_chd_all.csv'
FILE_SES_FEATURES = "/data/projects/project_GNN/AllIndFeat"
FILE_DRUG_FEATURES = "/data/projects/project_GNN/AgeInput/FullMat_Age.Drug"
FILE_ENDPOINT_FEATURES = "/data/projects/project_GNN/AgeInput/FullMat_Age.EndPt"

reader1 = pd.read_csv(FILE_SES_FEATURES, sep=',',chunksize=100_000, usecols=['FINREGISTRYID']+extra_SES) 
reader2 = pd.read_csv(FILE_DRUG_FEATURES, sep=',',chunksize=100_000, usecols=['FINREGISTRYID']+extra_Drug) 
reader3 = pd.read_csv(FILE_ENDPOINT_FEATURES, sep=',',chunksize=100_000, usecols=['FINREGISTRYID']+extra_EndPt) 

start = time.time()
for extra1,extra2,extra3 in zip(reader1,reader2,reader3):
    new_statfile = statfile.merge(extra1, how='inner',on='FINREGISTRYID')
    new_statfile = new_statfile.merge(extra2, how='inner',on='FINREGISTRYID')
    new_statfile = new_statfile.merge(extra3, how='inner',on='FINREGISTRYID')
    if not new_statfile.empty:
        new_statfile.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
print('added all previous features together into the dataset, time taken in seconds:')
print(time.time()-start)


#-----------------------------
print('start data sorting')

df = pd.read_csv(OUTPUT_FILE)
df = df.sort_values(by='node_id')
df.to_csv(OUTPUT_FILE,index=None)

#-----------------------------
print('start fix missing values in SES')

df = pd.read_csv(OUTPUT_FILE)
columns_to_fill = df.columns[df.isnull().sum()!=0] 
for column in columns_to_fill:
    median_value = df[column].median()
    df[column].fillna(median_value, inplace=True)
df.to_csv(OUTPUT_FILE,index=False)


gc.collect()
print('finished')