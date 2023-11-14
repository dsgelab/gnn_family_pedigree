# PREPARE ENVIRONMENT

# run this in ePouta: 
# source /data/projects/project_GNN/envs/graphml/bin/activate

import pandas as pd
import gc
import os
import time

# prepare files
print('FETCHING DATA')
FILE_EXTRA_FEATURES = "/data/projects/project_GNN/AllIndFeat"
statfile = pd.read_csv(
    filepath_or_buffer = '/data/projects/project_GNN/gnn_family_pedigree/data/statfile.csv', 
    usecols = ['FINREGISTRYID','node_id','birth_year','sex','I9_CHD_nEvent','CHD_binary'])
statfile = statfile[['FINREGISTRYID','node_id','birth_year','sex','I9_CHD_nEvent','CHD_binary']]

# add endpoint label
# LABEL_COLS = ['CHD_binary','T2D_binary','ASTHMA_binary','DEPRESSION_binary','COL_CANC_binary'] 

# download extra feature names
extra_SES = pd.read_csv("/data/projects/project_GNN/Indep0.2/SES",header=None)[0].tolist()
extra_Drug = pd.read_csv("/data/projects/project_GNN/Indep0.2/Drug",header=None)[0].tolist()
extra_EndPt = pd.read_csv("/data/projects/project_GNN/Indep0.2/EndPt",header=None)[0].tolist()

# update names
extra_Drug = [el+'_nEvent' for el in extra_Drug] 
extra_EndPt = [el+'_nEvent' for el in extra_EndPt] 
extra_all = extra_SES + extra_Drug + extra_EndPt 


#-----------------------------
print('prepearing estended statfile headers')

# Drug
OUTPUT_FILE = '/data/projects/project_GNN/gnn_family_pedigree/data/extended_corr02/statfile_Drug.csv'
output_columns = ['FINREGISTRYID','node_id','I9_CHD_nEvent','CHD_binary'] + extra_Drug 
output = pd.DataFrame(columns=output_columns)
output.to_csv(OUTPUT_FILE, mode='w', index=False)

# EndPt
OUTPUT_FILE = '/data/projects/project_GNN/gnn_family_pedigree/data/extended_corr02/statfile_EndPt.csv'
output_columns = ['FINREGISTRYID','node_id','I9_CHD_nEvent','CHD_binary'] + extra_EndPt 
output = pd.DataFrame(columns=output_columns)
output.to_csv(OUTPUT_FILE, mode='w', index=False)


# SES
OUTPUT_FILE = '/data/projects/project_GNN/gnn_family_pedigree/data/extended_corr02/statfile_SES.csv'
output_columns = ['FINREGISTRYID','node_id','birth_year','sex','I9_CHD_nEvent','CHD_binary'] + extra_SES
output_columns[output_columns.index('247_psychiatric_residential_care')] = 'psychiatric_residential_care'
output_columns[output_columns.index('247_residential_care_housing_under_65yo')] = 'residential_care_housing_under_65yo'
output = pd.DataFrame(columns=output_columns)
output.to_csv(OUTPUT_FILE, mode='w', index=False)

# all 3
OUTPUT_FILE = '/data/projects/project_GNN/gnn_family_pedigree/data/extended_corr02/statfile_all.csv'
output_columns = ['FINREGISTRYID','node_id','birth_year','sex','I9_CHD_nEvent','CHD_binary'] + extra_all
output_columns[output_columns.index('247_psychiatric_residential_care')] = 'psychiatric_residential_care'
output_columns[output_columns.index('247_residential_care_housing_under_65yo')] = 'residential_care_housing_under_65yo'
output = pd.DataFrame(columns=output_columns)
output.to_csv(OUTPUT_FILE, mode='w', index=False)

#-----------------------------
print('appending to extendended DRUG')

OUTPUT_FILE = '/data/projects/project_GNN/gnn_family_pedigree/data/extended_corr02/statfile_Drug.csv'
reader = pd.read_csv(FILE_EXTRA_FEATURES, sep=',',chunksize=100_000, usecols=['FINREGISTRYID']+extra_Drug) 
start = time.time()
for extra in reader:
    new_statfile = statfile.merge(extra, how='inner',on='FINREGISTRYID')
    if not new_statfile.empty:
        new_statfile = new_statfile.drop(columns=['birth_year','sex'])
        new_statfile.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

print('added {} medication features into the dataset'.format(len(extra.columns)-1) )
print(time.time()-start)

#-----------------------------
print('appending to extendended ENDPOINT')

OUTPUT_FILE = '/data/projects/project_GNN/gnn_family_pedigree/data/extended_corr02/statfile_EndPt.csv'
reader = pd.read_csv(FILE_EXTRA_FEATURES, sep=',',chunksize=100_000, usecols=['FINREGISTRYID']+extra_EndPt) 
start = time.time()
for extra in reader:
    new_statfile = statfile.merge(extra, how='inner',on='FINREGISTRYID')
    if not new_statfile.empty:
        new_statfile = new_statfile.drop(columns=['birth_year','sex'])
        new_statfile.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
        

print('added {} endpoint features into the dataset'.format(len(extra.columns)-1) )
print(time.time()-start)

#-----------------------------
print('appending to extendended SES')

OUTPUT_FILE = '/data/projects/project_GNN/gnn_family_pedigree/data/extended_corr02/statfile_SES.csv'
reader = pd.read_csv(FILE_EXTRA_FEATURES, sep=',',chunksize=100_000, usecols=['FINREGISTRYID']+extra_SES) 
start = time.time()
for extra in reader:
    new_statfile = statfile.merge(extra, how='inner',on='FINREGISTRYID')
    if not new_statfile.empty:
        new_statfile.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

print('added {} SES features into the dataset'.format(len(extra.columns)-1) )
print(time.time()-start)

#-----------------------------
print('appending to extendended ALL')

OUTPUT_FILE = '/data/projects/project_GNN/gnn_family_pedigree/data/extended_corr02/statfile_all.csv'
reader = pd.read_csv(FILE_EXTRA_FEATURES, sep=',',chunksize=100_000, usecols=['FINREGISTRYID']+extra_all) 
start = time.time()
for extra in reader:
    new_statfile = statfile.merge(extra, how='inner',on='FINREGISTRYID')
    if not new_statfile.empty:
        new_statfile.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

print('added all previous features together into the dataset')
print(time.time()-start)

#-----------------------------
print('start data sorting')

for name in ['Drug','EndPt','SES','all']:
    filename = "/data/projects/project_GNN/gnn_family_pedigree/data/extended_corr02/statfile_{}.csv".format(name)
    df = pd.read_csv(filename)
    df = df.sort_values(by='node_id')
    df.to_csv(filename,index=None)

#-----------------------------
print('start fix missing values in SES')

df = pd.read_csv('/data/projects/project_GNN/gnn_family_pedigree/data/extended_corr02/statfile_SES.csv')
columns_to_fill = df.columns[df.isnull().sum()!=0] 
for column in columns_to_fill:
    median_value = df[column].median()
    df[column].fillna(median_value, inplace=True)
df.to_csv('/data/projects/project_GNN/gnn_family_pedigree/data/extended_corr02/statfile_all.csv',index=False)


gc.collect()
print('finished')



